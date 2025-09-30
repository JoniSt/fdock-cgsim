#include "processligand.h"

#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <utility>
#include <type_traits>
#include <array>
#include <algorithm>
#include <bit>
#include <climits>

#include <boost/assert.hpp>

#include "json.hpp"
#include "graphtoy/graphtoy.hpp"
#include "cgsim/cgsim.hpp"

using namespace graphtoy;
using namespace ttlhacker::cgsim;

static constexpr auto s_iop_rtp = IoPortEndpointOptions{.m_isSingleWrite = true};

static constexpr auto g_maxNumAtoms = 256;
static constexpr auto g_maxNumAtomTypes = 14;
static constexpr auto g_numDistanceIDs = 2048;
static constexpr auto g_distanceIDStep = 0.01;
static constexpr auto g_desolvSigma = 3.6;
static constexpr auto g_outOfGridPenalty = 1 << 24;
static constexpr auto g_peratomOutOfGridPenalty = 100000;
static constexpr auto g_maxNumRotbonds = 32;
static constexpr auto g_numChangeConformRotateKernels = 4;

// The AIE's AXI4 FIFOs have a depth of four 32-bit values (16 bytes total).
static constexpr size_t g_aieAxiFifoDepthBytes = 16;

// FIFO depth that better emulates the AI Engines.
template<typename T>
static constexpr size_t g_fifoDepthFor = std::max(size_t(1), g_aieAxiFifoDepthBytes / sizeof(T));


static bool g_graphdumpsEnabled = false;
static const std::string g_dumpDirectory = "dumps";


static void forEachDistanceID(auto&& fn) {
    double dist = 0;

    for (int i = 0; i < g_numDistanceIDs; ++i) {
        dist += g_distanceIDStep;
        fn(i, dist);
    }
}


//using AtomIndexPair = std::pair<uint8_t, uint8_t>;

struct AtomIndexPair {
    uint8_t m_first;
    uint8_t m_second;
    bool m_terminate_processing;
};

struct IntraE_AtomPair {
    bool m_terminate_processing = false;
    bool m_isHBond = false;

    double m_atom1_idxyzq[5] = {};
    double m_atom2_idxyzq[5] = {};

    int m_distanceID = 0;

    double m_s1 = 0;
    double m_s2 = 0;
    double m_v1 = 0;
    double m_v2 = 0;

    double m_vdW1 = 0;
    double m_vdW2 = 0;
};


template<typename InputT, typename OutputT>
struct IntraE_ComputeKernelBase: GtKernelBase {
    explicit IntraE_ComputeKernelBase(GtContext *ctx): GtKernelBase(ctx) {}
        
    auto * input() { return m_inputStream; }
    auto * output() { return m_outputStream; }

protected:
    GtKernelIoStream<InputT> *m_inputStream = addIoStream<InputT>(g_fifoDepthFor<InputT>);
    GtKernelIoStream<OutputT> *m_outputStream = addIoStream<OutputT>(g_fifoDepthFor<OutputT>);
};

using IntraE_ChainKernel = IntraE_ComputeKernelBase<IntraE_AtomPair, IntraE_AtomPair>;

/**
 * Input: intraE_contributors array
 * Output: Stream of atom pairs (as indices) that must be considered for intraE
 */
COMPUTE_KERNEL(hls, kernel_IntraE_GenAtomPairIndices,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelMemoryPort<const char> intraE_contributors_buf,
    KernelWritePort<AtomIndexPair> atom_pair_out
) {
    uint32_t num_atoms = co_await num_atoms_in.get();

    num_atoms = std::min(num_atoms, uint32_t(g_maxNumAtoms));

    for (uint32_t x = 0; x < num_atoms; ++x) {
        for (uint32_t y = 0; y < num_atoms; ++y) {
            const char atomPairContributes = intraE_contributors_buf[x * g_maxNumAtoms + y];

            if (x < y && atomPairContributes) {
                co_await atom_pair_out.put(AtomIndexPair{uint8_t(x), uint8_t(y), false});
            }
        }
    }

    co_await atom_pair_out.put(AtomIndexPair{0, 0, true});
}

/**
 * Input: Stream of atom pairs (as indices)
 * Output: Stream of atom data (idxyzq)
 */
COMPUTE_KERNEL(hls, kernel_IntraE_FetchAtomData,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelReadPort<AtomIndexPair> atom_pair_in,
    KernelMemoryPort<const double> atom_idxyzq_buf,
    KernelWritePort<IntraE_AtomPair> atom_data_out
) {
    double atom_idxyzq[g_maxNumAtoms][5];

    uint32_t num_atoms = co_await num_atoms_in.get();
    num_atoms = std::min(num_atoms, uint32_t(g_maxNumAtoms));

    // Copy the entire atom_idxyzq array into local memory
    for (uint32_t idx_atom = 0; idx_atom < num_atoms; ++idx_atom) {
        for (size_t i = 0; i < 5; ++i) {
            atom_idxyzq[idx_atom][i] = atom_idxyzq_buf[idx_atom * 5 + i];
        }
    }

    // Read atom pairs
    while (true) {
        const auto index_pair = co_await atom_pair_in.get();
        if (index_pair.m_terminate_processing) {
            break;
        }

        IntraE_AtomPair result{};
        std::memcpy(result.m_atom1_idxyzq, atom_idxyzq[index_pair.m_first], sizeof(result.m_atom1_idxyzq));
        std::memcpy(result.m_atom2_idxyzq, atom_idxyzq[index_pair.m_second], sizeof(result.m_atom2_idxyzq));

        co_await atom_data_out.put(result);
    }

    // Terminate pipeline
    co_await atom_data_out.put(IntraE_AtomPair{.m_terminate_processing = true});
}

static std::vector<char> intraE_build_hbond_lut(const Liganddata* myligand) {
    const auto numAtomTypes = myligand->num_of_atypes;
    BOOST_ASSERT_MSG(numAtomTypes <= g_maxNumAtomTypes, "Invalid number of atom types");

    std::vector<char> is_hbond_lut(g_maxNumAtomTypes * g_maxNumAtomTypes, 0);

    for (int type_id1 = 0; type_id1 < numAtomTypes; ++type_id1) {
        for (int type_id2 = 0; type_id2 < numAtomTypes; ++type_id2) {
            is_hbond_lut[type_id1 * g_maxNumAtomTypes + type_id2] = is_H_bond(myligand->atom_types[type_id1], myligand->atom_types[type_id2]) != 0;
        }
    }

    return is_hbond_lut;
}

/**
 * Input: Stream of atom data
 * Output: Stream of atom data, with m_distanceID and m_isHBond set appropriately, filtered by distance
 */
COMPUTE_KERNEL(hls, kernel_IntraE_SetDistanceID_CheckHBond,
    KernelReadPort<IntraE_AtomPair> atom_data_in,
    KernelWritePort<IntraE_AtomPair> atom_data_out,
    KernelReadPort<double, s_iop_rtp> dcutoff_in,
    KernelMemoryPort<const char> is_hbond_lut_buf
) {
    const double dcutoff = co_await dcutoff_in.get();

    // Make a local copy of the H-bond LUT
    constexpr size_t num_hbond_lut_entries = g_maxNumAtomTypes * g_maxNumAtomTypes;
    char is_hbond_lut[num_hbond_lut_entries];

    for (size_t i = 0; i < num_hbond_lut_entries; ++i) {
        is_hbond_lut[i] = is_hbond_lut_buf[i];
    }

    while (true) {
        auto data = co_await atom_data_in.get();

        if (!data.m_terminate_processing) {
            double dist = distance(&(data.m_atom1_idxyzq[1]), &(data.m_atom2_idxyzq[1]));
            dist = std::max(dist, 1.0);

            auto& distance_id = data.m_distanceID;
            distance_id = static_cast<int>(std::floor((100 * dist) + 0.5)) - 1; // +0.5: rounding, -1: r_xx_table [0] corresponds to r=0.01
            distance_id = std::max(distance_id, 0);

            if (dist >= dcutoff || distance_id >= g_numDistanceIDs) {
                continue;
            }

            int type_id1 = static_cast<int>(data.m_atom1_idxyzq[0]);
            int type_id2 = static_cast<int>(data.m_atom2_idxyzq[0]);

            data.m_isHBond = is_hbond_lut[type_id1 * g_maxNumAtomTypes + type_id2];
        }

        co_await atom_data_out.put(data);

        if (data.m_terminate_processing) break;
    }
}

/**
 * Populates s1, s2, v1, v2 in the atom data that passes through.
 */
COMPUTE_KERNEL(hls, kernel_IntraE_Volume_Solpar,
    KernelReadPort<IntraE_AtomPair> atom_data_in,
    KernelWritePort<IntraE_AtomPair> atom_data_out,
    KernelReadPort<double, s_iop_rtp> qasp_in,
    KernelMemoryPort<const double> volume_buf,
    KernelMemoryPort<const double> solpar_buf
) {
    const double qasp = co_await qasp_in.get();

    double volume[g_maxNumAtomTypes];
    double solpar[g_maxNumAtomTypes];

    for (size_t i = 0; i < g_maxNumAtomTypes; ++i) {
        volume[i] = volume_buf[i];
        solpar[i] = solpar_buf[i];
    }

    while (true) {
        auto data = co_await atom_data_in.get();

        if (!data.m_terminate_processing) {
            int type_id1 = static_cast<int>(data.m_atom1_idxyzq[0]);
            int type_id2 = static_cast<int>(data.m_atom2_idxyzq[0]);

            double q1 = data.m_atom1_idxyzq[4];
            double q2 = data.m_atom2_idxyzq[4];

            data.m_s1 = solpar[type_id1] + qasp * std::fabs(q1);
            data.m_s2 = solpar[type_id2] + qasp * std::fabs(q2);
            data.m_v1 = volume[type_id1];
            data.m_v2 = volume[type_id2];
        }

        co_await atom_data_out.put(data);

        if (data.m_terminate_processing) break;
    }
}

/**
 * Populates m_vdW1 and m_vdW2 in the atom data that passes through with the appropriate entries from VWpars_x.
 * Does not perform multiplication with r**{6,10,12} yet.
 */
COMPUTE_KERNEL(hls, kernel_IntraE_FetchVWpars,
    KernelReadPort<IntraE_AtomPair> atom_data_in,
    KernelWritePort<IntraE_AtomPair> atom_data_out,
    KernelMemoryPort<const double> vwpars_a_buf,
    KernelMemoryPort<const double> vwpars_b_buf,
    KernelMemoryPort<const double> vwpars_c_buf,
    KernelMemoryPort<const double> vwpars_d_buf
) {
    double vwpars_a[g_maxNumAtomTypes][g_maxNumAtomTypes];
    double vwpars_b[g_maxNumAtomTypes][g_maxNumAtomTypes];
    double vwpars_c[g_maxNumAtomTypes][g_maxNumAtomTypes];
    double vwpars_d[g_maxNumAtomTypes][g_maxNumAtomTypes];

    for (size_t i = 0; i < g_maxNumAtomTypes; ++i) {
        for (size_t j = 0; j < g_maxNumAtomTypes; ++j) {
            vwpars_a[i][j] = vwpars_a_buf[i * g_maxNumAtomTypes + j];
            vwpars_b[i][j] = vwpars_b_buf[i * g_maxNumAtomTypes + j];
            vwpars_c[i][j] = vwpars_c_buf[i * g_maxNumAtomTypes + j];
            vwpars_d[i][j] = vwpars_d_buf[i * g_maxNumAtomTypes + j];
        }
    }

    while (true) {
        auto data = co_await atom_data_in.get();

        if (!data.m_terminate_processing) {
            int type_id1 = static_cast<int>(data.m_atom1_idxyzq[0]);
            int type_id2 = static_cast<int>(data.m_atom2_idxyzq[0]);

            if (data.m_isHBond) {
                data.m_vdW1 = vwpars_c[type_id1][type_id2];
                data.m_vdW2 = vwpars_d[type_id1][type_id2];
            } else {
                data.m_vdW1 = vwpars_a[type_id1][type_id2];
                data.m_vdW2 = vwpars_b[type_id1][type_id2];
            }
        }

        co_await atom_data_out.put(data);

        if (data.m_terminate_processing) break;
    }
}

struct IntraE_DistanceLuts {
    // 24KB total when using single-precision floats
    double m_r_6_table[g_numDistanceIDs];
    double m_r_10_table[g_numDistanceIDs];
    double m_r_12_table[g_numDistanceIDs];

    // 16KB total with single-precision floats
    double m_r_epsr_table_unscaled[g_numDistanceIDs];
    double m_desolv_table_unscaled[g_numDistanceIDs];

    IntraE_DistanceLuts() {
        forEachDistanceID([this](int i, double dist) {
            m_r_6_table[i] = 1/pow(dist,6);
            m_r_10_table[i] = 1/pow(dist,10);
            m_r_12_table[i] = 1/pow(dist,12);

            m_r_epsr_table_unscaled[i] = dist*calc_ddd_Mehler_Solmajer(dist);
            m_desolv_table_unscaled[i] = exp(-1*dist*dist/(2*g_desolvSigma*g_desolvSigma));
        });
    }
};

static const IntraE_DistanceLuts g_intraE_luts{};

/**
 * Multiplies m_vdW1 and m_vdW2 with the appropriate power of the distance.
 */
COMPUTE_KERNEL(hls, kernel_IntraE_ScaleVWparsWithDistance,
    KernelReadPort<IntraE_AtomPair> atom_data_in,
    KernelWritePort<IntraE_AtomPair> atom_data_out
) {
    while (true) {
        auto data = co_await atom_data_in.get();

        if (!data.m_terminate_processing) {
            const auto did = data.m_distanceID;

            data.m_vdW1 *= g_intraE_luts.m_r_12_table[did];
            data.m_vdW2 *= data.m_isHBond ? g_intraE_luts.m_r_10_table[did] : g_intraE_luts.m_r_6_table[did];
        }

        co_await atom_data_out.put(data);

        if (data.m_terminate_processing) break;
    }
}

/**
 * Computes the final energies and accumulates them for all atoms
 */
COMPUTE_KERNEL(hls, kernel_IntraE_Compute_VW_EL_Desolv,
    KernelReadPort<IntraE_AtomPair> atom_data_in,
    KernelReadPort<double, s_iop_rtp> scaled_AD4_coeff_elec_in,
    KernelReadPort<double, s_iop_rtp> AD4_coeff_desolv_in,
    KernelWritePort<double, s_iop_rtp> vW_out,
    KernelWritePort<double, s_iop_rtp> el_out,
    KernelWritePort<double, s_iop_rtp> desolv_out
) {
    double vW = 0;
    double el = 0;
    double desolv = 0;

    const double epsrScale = co_await scaled_AD4_coeff_elec_in.get();
    const double desolvScale = co_await AD4_coeff_desolv_in.get();

    while (true) {
        const auto data = co_await atom_data_in.get();

        if (data.m_terminate_processing) {
            break;
        }

        const auto q1 = data.m_atom1_idxyzq[4];
        const auto q2 = data.m_atom2_idxyzq[4];

        vW += data.m_vdW1 - data.m_vdW2;
        el += q1 * q2 * (epsrScale / g_intraE_luts.m_r_epsr_table_unscaled[data.m_distanceID]);
        desolv += (data.m_s1 * data.m_v2 + data.m_s2 * data.m_v1) * (desolvScale * g_intraE_luts.m_desolv_table_unscaled[data.m_distanceID]);
    }

    co_await vW_out.put(vW);
    co_await el_out.put(el);
    co_await desolv_out.put(desolv);
}

COMPUTE_GRAPH constexpr auto intraE_graph = make_compute_graph_v<[] (
    IoConnector<uint32_t> num_atoms_in,
    IoConnector<const char> intraE_contributors_buf,
    IoConnector<const double> atom_idxyzq_buf,
    IoConnector<const char> is_hbond_lut_buf,
    IoConnector<const double> volume_buf,
    IoConnector<const double> solpar_buf,
    IoConnector<const double> vwpars_a_buf,
    IoConnector<const double> vwpars_b_buf,
    IoConnector<const double> vwpars_c_buf,
    IoConnector<const double> vwpars_d_buf,
    IoConnector<double> dcutoff_in,
    IoConnector<double> qasp_in,
    IoConnector<double> scaled_AD4_coeff_elec_in,
    IoConnector<double> AD4_coeff_desolv_in
) {
    IoConnector<AtomIndexPair> atom_pairs;
    IoConnector<IntraE_AtomPair>
        atom_data_fetched,
        atom_data_distance_checked,
        atom_data_with_volume,
        atom_data_with_vw_fetched,
        atom_data_with_vw_scaled;

    IoConnector<double> vW_out, el_out, desolv_out;

    CGSIM_AUTO_NAME(num_atoms_in);
    CGSIM_AUTO_NAME(intraE_contributors_buf);
    CGSIM_AUTO_NAME(atom_idxyzq_buf);
    CGSIM_AUTO_NAME(volume_buf);
    CGSIM_AUTO_NAME(solpar_buf);
    CGSIM_AUTO_NAME(vwpars_a_buf);
    CGSIM_AUTO_NAME(vwpars_b_buf);
    CGSIM_AUTO_NAME(vwpars_c_buf);
    CGSIM_AUTO_NAME(vwpars_d_buf);
    CGSIM_AUTO_NAME(dcutoff_in);
    CGSIM_AUTO_NAME(qasp_in);
    CGSIM_AUTO_NAME(scaled_AD4_coeff_elec_in);
    CGSIM_AUTO_NAME(AD4_coeff_desolv_in);
    CGSIM_AUTO_NAME(vW_out);
    CGSIM_AUTO_NAME(el_out);
    CGSIM_AUTO_NAME(desolv_out);

    kernel_IntraE_GenAtomPairIndices(num_atoms_in, intraE_contributors_buf, atom_pairs);
    kernel_IntraE_FetchAtomData(num_atoms_in, atom_pairs, atom_idxyzq_buf, atom_data_fetched);
    kernel_IntraE_SetDistanceID_CheckHBond(atom_data_fetched, atom_data_distance_checked, dcutoff_in, is_hbond_lut_buf);
    kernel_IntraE_Volume_Solpar(atom_data_distance_checked, atom_data_with_volume, qasp_in, volume_buf, solpar_buf);
    kernel_IntraE_FetchVWpars(atom_data_with_volume, atom_data_with_vw_fetched, vwpars_a_buf, vwpars_b_buf, vwpars_c_buf, vwpars_d_buf);
    kernel_IntraE_ScaleVWparsWithDistance(atom_data_with_vw_fetched, atom_data_with_vw_scaled);
    kernel_IntraE_Compute_VW_EL_Desolv(atom_data_with_vw_scaled, scaled_AD4_coeff_elec_in, AD4_coeff_desolv_in, vW_out, el_out, desolv_out);

    return std::tuple(vW_out, el_out, desolv_out);
}>;


double calc_intraE_graphtoy(const Liganddata* myligand, double dcutoff, char ignore_desolv, const double scaled_AD4_coeff_elec, const double AD4_coeff_desolv, const double qasp) {
    // Copy all 2D arrays into local (flat) buffers
    static_assert(sizeof(myligand->intraE_contributors) == (g_maxNumAtoms * g_maxNumAtoms));
    std::vector<char> intraE_contributors_buf(g_maxNumAtoms * g_maxNumAtoms);
    std::memcpy(intraE_contributors_buf.data(), myligand->intraE_contributors, sizeof(myligand->intraE_contributors));

    static_assert(sizeof(myligand->atom_idxyzq) == (g_maxNumAtoms * 5 * sizeof(double)));
    std::vector<double> atom_idxyzq_buf(g_maxNumAtoms * 5);
    std::memcpy(atom_idxyzq_buf.data(), myligand->atom_idxyzq, sizeof(myligand->atom_idxyzq));

    static_assert(sizeof(myligand->VWpars_A) == (g_maxNumAtomTypes * g_maxNumAtomTypes * sizeof(double)));
    std::vector<double> vwpars_a_buf(g_maxNumAtomTypes * g_maxNumAtomTypes);
    std::memcpy(vwpars_a_buf.data(), myligand->VWpars_A, sizeof(myligand->VWpars_A));

    static_assert(sizeof(myligand->VWpars_B) == (g_maxNumAtomTypes * g_maxNumAtomTypes * sizeof(double)));
    std::vector<double> vwpars_b_buf(g_maxNumAtomTypes * g_maxNumAtomTypes);
    std::memcpy(vwpars_b_buf.data(), myligand->VWpars_B, sizeof(myligand->VWpars_B));

    static_assert(sizeof(myligand->VWpars_C) == (g_maxNumAtomTypes * g_maxNumAtomTypes * sizeof(double)));
    std::vector<double> vwpars_c_buf(g_maxNumAtomTypes * g_maxNumAtomTypes);
    std::memcpy(vwpars_c_buf.data(), myligand->VWpars_C, sizeof(myligand->VWpars_C));

    static_assert(sizeof(myligand->VWpars_D) == (g_maxNumAtomTypes * g_maxNumAtomTypes * sizeof(double)));
    std::vector<double> vwpars_d_buf(g_maxNumAtomTypes * g_maxNumAtomTypes);
    std::memcpy(vwpars_d_buf.data(), myligand->VWpars_D, sizeof(myligand->VWpars_D));

    // HBond LUT
    auto is_hbond_lut_buf = intraE_build_hbond_lut(myligand);

    double vW = 0;
    double el = 0;
    double desolv = 0;

    // Run graph
    const auto result = intraE_graph(
        ScalarDataSource<uint32_t>(myligand->num_of_atoms),
        memBuffer(intraE_contributors_buf),
        memBuffer(atom_idxyzq_buf),
        memBuffer(is_hbond_lut_buf),
        RuntimeMemoryBuffer(std::span<const double>(myligand->volume, g_maxNumAtomTypes)),
        RuntimeMemoryBuffer(std::span<const double>(myligand->solpar, g_maxNumAtomTypes)),
        memBuffer(vwpars_a_buf),
        memBuffer(vwpars_b_buf),
        memBuffer(vwpars_c_buf),
        memBuffer(vwpars_d_buf),
        ScalarDataSource<double>(dcutoff),
        ScalarDataSource<double>(qasp),
        ScalarDataSource<double>(scaled_AD4_coeff_elec),
        ScalarDataSource<double>(AD4_coeff_desolv),
        ScalarDataSink<double>(vW),
        ScalarDataSink<double>(el),
        ScalarDataSink<double>(desolv)
    );

    // Warn about deadlocks
    result.dump(std::cerr);

    return vW + el + (ignore_desolv ? 0.0 : desolv);
}

static void dumpStructRaw(const char *fileName, const char *data, size_t size) {
    std::ofstream stream{fileName, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary};
    if (!stream.good()) return;

    stream.write(data, size);
}

static void dumpLigand(const char *fileName, const Liganddata *ligand) {
    dumpStructRaw(fileName, (const char *)ligand, sizeof(*ligand));
}

static void dumpJson(const char *fileName, const nlohmann::json& json) {
    std::string s = json.dump(4);
    dumpStructRaw(fileName, s.data(), s.size());
}

double calc_intraE(const Liganddata* myligand, double dcutoff, char ignore_desolv, const double scaled_AD4_coeff_elec, const double AD4_coeff_desolv, const double qasp, int debug) {
    const double originalResult = calc_intraE_original(myligand, dcutoff, ignore_desolv, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp, debug);
    const double graphResult    = calc_intraE_graphtoy(myligand, dcutoff, ignore_desolv, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp);

    if (graphResult != originalResult) {
        std::cerr << "IntraE mismatch: original=" << originalResult << ", graph=" << graphResult << "\n";
    }

    if (g_graphdumpsEnabled) {
        static uint32_t s_dumpIndex = 0;
        std::string baseName = g_dumpDirectory + "/intraE_" + std::to_string(s_dumpIndex++);

        dumpLigand((baseName + "_ligand.bin").data(), myligand);

        nlohmann::json params{};
        params["dcutoff"] = dcutoff;
        params["ignore_desolv"] = bool(ignore_desolv);
        params["scaled_AD4_coeff_elec"] = scaled_AD4_coeff_elec;
        params["AD4_coeff_desolv"] = AD4_coeff_desolv;
        params["qasp"] = qasp;
        params["result"] = graphResult;
        dumpJson((baseName + "_params.json").data(), params);
    }

    return graphResult;
}


using InterE_RawAtomInput = double[5];

struct InterE_AtomInput {
    InterE_RawAtomInput m_atom_idxyzq = {};
    
    InterE_AtomInput() = default;
    
    /* conversion */ InterE_AtomInput(const InterE_RawAtomInput& data) {
        std::memcpy(m_atom_idxyzq, data, sizeof(m_atom_idxyzq));
    }
};

struct InterE_AtomData {
    int m_atom_cnt = 0;
    int m_type_id  = 0;
    
    bool m_isOutOfGrid = false;
    
    int m_x_low  = 0;
    int m_x_high = 0;
    int m_y_low  = 0;
    int m_y_high = 0;
    int m_z_low  = 0;
    int m_z_high = 0;
    
    double m_q = 0;
    
    double m_weights[2][2][2] = {};
};

struct InterE_AtomEnergy {
    int m_atom_cnt = 0;
    bool m_isOutOfGrid = false;
    
    double m_atomTypeGridEnergy = 0;
    double m_electrostaticGridEnergy = 0;
    double m_desolvationGridEnergy = 0;
};


using InterE_ChainKernel = IntraE_ComputeKernelBase<InterE_AtomData, InterE_AtomData>;

static bool interE_nudgeGridCoordsIntoBounds(double& x, double& y, double& z, const double outofgrid_tolerance, const std::array<int, 3>& size_xyz) {
    const auto isOutOfGrid = [&] {
        return (x < 0) || (x >= size_xyz [0]-1) ||
               (y < 0) || (y >= size_xyz [1]-1) ||
               (z < 0) || (z >= size_xyz [2]-1);
    };
    
    if (x < 0)
        x += outofgrid_tolerance;
    if (y < 0)
        y += outofgrid_tolerance;
    if (z < 0)
        z += outofgrid_tolerance;
    if (x >= size_xyz [0]-1)
        x -= outofgrid_tolerance;
    if (y >= size_xyz [1]-1)
        y -= outofgrid_tolerance;
    if (z >= size_xyz [2]-1)
        z -= outofgrid_tolerance;
    
    return isOutOfGrid();
}

static int interE_gridNumberToArrayOffset(const std::array<int, 3>& size_xyz, int t) {
    return t * size_xyz[0] * size_xyz[1] * size_xyz[2];
}

static int interE_gridCoordsToArrayOffset(const std::array<int, 3>& size_xyz, int z, int y, int x) {
    return x + size_xyz[0] * (y + size_xyz[1] * z);
}

struct Kernel_InterE_BuildAtomData: IntraE_ComputeKernelBase<InterE_AtomInput, InterE_AtomData> {
    Kernel_InterE_BuildAtomData(GtContext *ctx, double outofgrid_tolerance, std::array<int, 3> size_xyz):
        IntraE_ComputeKernelBase<InterE_AtomInput, InterE_AtomData>(ctx),
        m_outofgrid_tolerance(outofgrid_tolerance), m_size_xyz(size_xyz) {}
    
private:
    double m_outofgrid_tolerance;
    std::array<int, 3> m_size_xyz;
    
    GtKernelCoro kernelMain() override {
        int atom_cnt = 0;
        while (true) {
            const InterE_AtomInput inputData = co_await m_inputStream->read();
            
            InterE_AtomData outputData{};
            outputData.m_atom_cnt = atom_cnt++;
            outputData.m_type_id = int(inputData.m_atom_idxyzq[0]);
            
            double x = inputData.m_atom_idxyzq[1];
            double y = inputData.m_atom_idxyzq[2];
            double z = inputData.m_atom_idxyzq[3];
            
            outputData.m_isOutOfGrid = interE_nudgeGridCoordsIntoBounds(x, y, z, m_outofgrid_tolerance, m_size_xyz);
            
            if (!outputData.m_isOutOfGrid) {
                outputData.m_q = inputData.m_atom_idxyzq[4];
                
                outputData.m_x_low = (int) floor(x);
                outputData.m_y_low = (int) floor(y);
                outputData.m_z_low = (int) floor(z);
                outputData.m_x_high = (int) ceil(x);
                outputData.m_y_high = (int) ceil(y);
                outputData.m_z_high = (int) ceil(z);
                
                const double x_frac = x - outputData.m_x_low;
                const double y_frac = y - outputData.m_y_low;
                const double z_frac = z - outputData.m_z_low;
                
                get_trilininterpol_weights(outputData.m_weights, x_frac, y_frac, z_frac);
            }
            
            co_await m_outputStream->write(outputData);
        }
    }
};

struct Kernel_InterE_GenerateDramAddresses: InterE_ChainKernel {
    Kernel_InterE_GenerateDramAddresses(GtContext *ctx, std::array<int, 3> size_xyz, int num_of_atypes):
        InterE_ChainKernel(ctx), m_size_xyz(size_xyz), m_num_of_atypes(num_of_atypes) {}
    
    auto * addressOutput() { return m_addressOutput; }
    
private:
    std::array<int, 3> m_size_xyz;
    int m_num_of_atypes;
    
    GtKernelIoStream<uint32_t> *m_addressOutput = addIoStream<uint32_t>(g_fifoDepthFor<uint32_t>);
    
    GtKernelCoro kernelMain() override {
        while (true) {
            const auto data = co_await m_inputStream->read();
            
            // The atom data must be written before its associated DRAM addresses because otherwise we'd
            // saturate the system's buffers and cause a deadlock. (This kernel would wait for more space
            // in the DRAM addr/data buffers before it writes the atom data, but the InterpolateEnergy
            // kernel would wait for the atom data before it drains the DRAM buffers)
            co_await m_outputStream->write(data);
            
            if (data.m_isOutOfGrid)
                continue;
            
            std::array<int, 8> coordOffsets = {
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_low, data.m_y_low, data.m_x_low),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_low, data.m_y_low, data.m_x_high),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_low, data.m_y_high, data.m_x_low),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_low, data.m_y_high, data.m_x_high),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_high, data.m_y_low, data.m_x_low),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_high, data.m_y_low, data.m_x_high),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_high, data.m_y_high, data.m_x_low),
                interE_gridCoordsToArrayOffset(m_size_xyz, data.m_z_high, data.m_y_high, data.m_x_high)
            };
            
            std::array<int, 3> gridNumbers = {
                data.m_type_id,     // energy contribution of the current grid type
                m_num_of_atypes,    // energy contribution of the electrostatic grid
                m_num_of_atypes + 1 // energy contribution of the desolvation grid
            };
            
            for (const int gridNumber: gridNumbers) {
                const int gridOffset = interE_gridNumberToArrayOffset(m_size_xyz, gridNumber);
                for (const int coordOffset: coordOffsets) {
                    const int finalAddr = gridOffset + coordOffset;
                    BOOST_ASSERT(finalAddr >= 0);
                    co_await m_addressOutput->write(uint32_t(finalAddr));
                }
            }
        }
    }
};

struct Kernel_InterE_InterpolateEnergy: IntraE_ComputeKernelBase<InterE_AtomData, InterE_AtomEnergy> {
    explicit Kernel_InterE_InterpolateEnergy(GtContext *ctx):
        IntraE_ComputeKernelBase<InterE_AtomData, InterE_AtomEnergy>(ctx) {}
        
    auto * dramDataInput() { return m_dramDataInput; }
    
private:
    GtKernelIoStream<double> *m_dramDataInput = addIoStream<double>(g_fifoDepthFor<double>);
    
    GtKernelCoro kernelMain() override {
        enum class GridType { ATOM, ELECTROSTATIC, DESOLVATION };
        using enum GridType;
        
        const auto getCubeElem = [&] { return m_dramDataInput->read(); };
        
        while (true) {
            const auto data = co_await m_inputStream->read();
            InterE_AtomEnergy result{.m_atom_cnt = data.m_atom_cnt, .m_isOutOfGrid = data.m_isOutOfGrid};
            
            if (!data.m_isOutOfGrid) {
                for (const GridType t: {ATOM, ELECTROSTATIC, DESOLVATION}) {
                    double cube[2][2][2];
                    
                    cube [0][0][0] = co_await getCubeElem();
                    cube [1][0][0] = co_await getCubeElem();
                    cube [0][1][0] = co_await getCubeElem();
                    cube [1][1][0] = co_await getCubeElem();
                    cube [0][0][1] = co_await getCubeElem();
                    cube [1][0][1] = co_await getCubeElem();
                    cube [0][1][1] = co_await getCubeElem();
                    cube [1][1][1] = co_await getCubeElem();
                    
                    const double interpolated = trilin_interpol(cube, data.m_weights);
                    
                    switch (t) {
                        case ATOM:              result.m_atomTypeGridEnergy         = interpolated;                     break;
                        case ELECTROSTATIC:     result.m_electrostaticGridEnergy    = data.m_q * interpolated;          break;
                        case DESOLVATION:       result.m_desolvationGridEnergy      = fabs(data.m_q) * interpolated;    break;
                    }
                }
            }
            
            co_await m_outputStream->write(result);
        }
    }
};

struct Kernel_InterE_AccumulateResults: GtKernelBase {
    Kernel_InterE_AccumulateResults(GtContext *ctx, bool enablePeratomOutputs):
        GtKernelBase(ctx)
    {
        if (enablePeratomOutputs) {
            m_vdwStream = addIoStream<double>(g_fifoDepthFor<double>);
            m_elecStream = addIoStream<double>(g_fifoDepthFor<double>);
        }
    }
    
    auto * input() { return m_inputStream; }
    
    auto getEnergy() { return m_interE; }
    auto getElecE()  { return m_elecE;  }
    
    auto * vdwOutput()  { return m_vdwStream;  }
    auto * elecOutput() { return m_elecStream; }
    
private:
    double m_interE = 0;
    double m_elecE = 0;
    
    GtKernelIoStream<InterE_AtomEnergy> *m_inputStream = addIoStream<InterE_AtomEnergy>(g_fifoDepthFor<InterE_AtomEnergy>);
    GtKernelIoStream<double> *m_vdwStream = nullptr;
    GtKernelIoStream<double> *m_elecStream = nullptr;
    
    GtKernelCoro kernelMain() override {
        while (true) {
            const auto data = co_await m_inputStream->read();
            
            auto vdW  = data.m_atomTypeGridEnergy;
            auto elec = data.m_electrostaticGridEnergy;
            
            m_interE += vdW;
            m_interE += elec;
            m_interE += data.m_desolvationGridEnergy;
            
            m_elecE += elec;
            
            if (data.m_isOutOfGrid) {
                m_interE += g_outOfGridPenalty;
                elec = vdW = g_peratomOutOfGridPenalty;
            }
            
            if (m_vdwStream)
                co_await m_vdwStream->write(vdW);
            
            if (m_elecStream)
                co_await m_elecStream->write(elec);
        }
    }
};


struct InterE_Result {
    double m_interE = 0;
    double m_elecE = 0;
    
    std::vector<double> m_peratomVdW{};
    std::vector<double> m_peratomElec{};
};

static auto getNumGridElems(const Gridinfo *mygrid) {
    const auto& gridsize = mygrid->size_xyz;
    return (mygrid->num_of_atypes + 2) * gridsize[0] * gridsize[1] * gridsize[2];
}

InterE_Result calc_interE_graphtoy(const Gridinfo* mygrid, const Liganddata* myligand, const double* fgrids, double outofgrid_tolerance, bool enablePerAtomOutputs) {
    GtContext ctx{};
    InterE_Result result{};
    
    BOOST_ASSERT(mygrid->num_of_atypes == myligand->num_of_atypes);
    
    std::array<int, 3> gridsize{};
    std::copy_n(std::begin(mygrid->size_xyz), 3, std::begin(gridsize));
    
    std::span<const InterE_RawAtomInput> inputAtoms{&myligand->atom_idxyzq[0], size_t(myligand->num_of_atoms)};
    
    const auto numGridMemElems = getNumGridElems(mygrid);
    std::span<const double> gridMemoryRegion{fgrids, size_t(numGridMemElems)};
    
    auto& atomInputStream   = ctx.addKernel<GtMemStreamSource<InterE_RawAtomInput, InterE_AtomInput>>(inputAtoms);
    auto& buildAtomData     = ctx.addKernel<Kernel_InterE_BuildAtomData>(outofgrid_tolerance, gridsize);
    auto& genDramAddrs      = ctx.addKernel<Kernel_InterE_GenerateDramAddresses>(gridsize, myligand->num_of_atypes);
    auto& fetchDramValues   = ctx.addKernel<GtFpgaDmaMemReader<double>>(gridMemoryRegion);
    auto& computeEnergy     = ctx.addKernel<Kernel_InterE_InterpolateEnergy>();
    auto& accumulateEnergy  = ctx.addKernel<Kernel_InterE_AccumulateResults>(enablePerAtomOutputs);
    
    GtMemStreamSink<double> *storePerAtomVdW  = nullptr;
    GtMemStreamSink<double> *storePerAtomElec = nullptr;
    
    // atoms -> buildAtomData -> genDramAddrs -----> computeEnergy -> accumulateEnergy -> accumulator
    //                                |                    ^                |
    //                                `-> fetchDramValues -'                |-> peratom vdW
    //                                                                      `-> peratom elec
    
    ctx.connect(atomInputStream.output(), buildAtomData.input());
    ctx.connect(buildAtomData.output(), genDramAddrs.input());
    ctx.connect(genDramAddrs.output(), computeEnergy.input());
    ctx.connect(genDramAddrs.addressOutput(), fetchDramValues.input());
    ctx.connect(fetchDramValues.output(), computeEnergy.dramDataInput());
    ctx.connect(computeEnergy.output(), accumulateEnergy.input());
    
    if (enablePerAtomOutputs) {
        storePerAtomVdW  = &ctx.addKernel<GtMemStreamSink<double>>();
        storePerAtomElec = &ctx.addKernel<GtMemStreamSink<double>>();
        ctx.connect(accumulateEnergy.vdwOutput(),  storePerAtomVdW->input());
        ctx.connect(accumulateEnergy.elecOutput(), storePerAtomElec->input());
    }
    
    ctx.runToCompletion();
    
    result.m_interE = accumulateEnergy.getEnergy();
    result.m_elecE  = accumulateEnergy.getElecE();
    
    if (enablePerAtomOutputs) {
        result.m_peratomVdW  = std::move(storePerAtomVdW->data());
        result.m_peratomElec = std::move(storePerAtomElec->data());
    }
    
    return result;
}

double calc_interE(const Gridinfo* mygrid, const Liganddata* myligand, const double* fgrids, double outofgrid_tolerance, int debug) {
    const double originalResult = calc_interE_original(mygrid, myligand, fgrids, outofgrid_tolerance, debug);
    const double graphResult = calc_interE_graphtoy(mygrid, myligand, fgrids, outofgrid_tolerance, false).m_interE;
    
    BOOST_ASSERT(graphResult == originalResult);
    
    if (g_graphdumpsEnabled) {
        static uint32_t s_dumpIndex = 0;
        const std::string baseNameNoIndex = g_dumpDirectory + "/interE_";
        const std::string baseName = baseNameNoIndex + std::to_string(s_dumpIndex++);

        dumpLigand((baseName + "_ligand.bin").data(), myligand);

        static bool s_gridDumped = false;
        if (!s_gridDumped) {
            s_gridDumped = true;
            const auto numGridMemElems = getNumGridElems(mygrid);
            dumpStructRaw((baseNameNoIndex + "grid.bin").data(), (const char *)fgrids, numGridMemElems * sizeof(fgrids[0]));
        }

        const auto& gridsize = mygrid->size_xyz;

        nlohmann::json params{};
        params["outofgrid_tolerance"] = outofgrid_tolerance;
        params["gridsize_x"] = gridsize[0];
        params["gridsize_y"] = gridsize[1];
        params["gridsize_z"] = gridsize[2];
        params["result"] = graphResult;
        dumpJson((baseName + "_params.json").data(), params);
    }
    
    return graphResult;
}

void calc_interE_peratom(const Gridinfo* mygrid, const Liganddata* myligand, const double* fgrids, double outofgrid_tolerance,
                         double* elecE, double peratom_vdw [256], double peratom_elec [256], int debug)
{
    calc_interE_peratom_original(mygrid, myligand, fgrids, outofgrid_tolerance, elecE, peratom_vdw, peratom_elec, debug);
    const auto graphResult = calc_interE_graphtoy(mygrid, myligand, fgrids, outofgrid_tolerance, true);
    
    const auto num_atoms = size_t(myligand->num_of_atoms);
    
    BOOST_ASSERT(graphResult.m_peratomVdW.size() == num_atoms);
    BOOST_ASSERT(graphResult.m_peratomElec.size() == num_atoms);
    
    const auto memeq = [](const std::vector<double>& lhs, const double *rhs) {
        return std::memcmp(lhs.data(), rhs, lhs.size() * sizeof(lhs[0])) == 0;
    };
    
    BOOST_ASSERT(memeq(graphResult.m_peratomVdW, peratom_vdw) && memeq(graphResult.m_peratomElec, peratom_elec));
    BOOST_ASSERT(graphResult.m_elecE == *elecE);
}


struct ChangeConform_AtomData {
    InterE_AtomInput m_atomdata;
    uint32_t m_rotbondMask;
    uint32_t m_numRotbondsPerStage;
    
    static_assert(sizeof(m_rotbondMask) * CHAR_BIT >= g_maxNumRotbonds);
};

using ChangeConform_ChainKernel = IntraE_ComputeKernelBase<ChangeConform_AtomData, ChangeConform_AtomData>;

static void vec3_accum(double *vec_a, const double *vec_b) {
    for (size_t i = 0; i < 3; ++i) {
        vec_a[i] += vec_b[i];
    }
}

/**
 * Performs up to m_numRotbondsPerStage pending rotations on the atom data passing through.
 * Multiple of these kernels must be chained together to perform all rotations for each atom.
 * This enables pipeline parallelism for ligand rotations.
 */
struct Kernel_ChangeConform_RotatePartial : ChangeConform_ChainKernel {
    Kernel_ChangeConform_RotatePartial(GtContext *ctx, const Liganddata *myligand, const double *genotype_rotbonds):
        ChangeConform_ChainKernel(ctx)
    {
        static_assert(sizeof(m_rotbonds_moving_vectors) == sizeof(myligand->rotbonds_moving_vectors));
        static_assert(sizeof(m_rotbonds_unit_vectors) == sizeof(myligand->rotbonds_unit_vectors));
        std::memcpy(m_rotbonds_moving_vectors, myligand->rotbonds_moving_vectors, sizeof(m_rotbonds_moving_vectors));
        std::memcpy(m_rotbonds_unit_vectors, myligand->rotbonds_unit_vectors, sizeof(m_rotbonds_unit_vectors));
        
        BOOST_ASSERT(myligand->num_of_rotbonds <= g_maxNumRotbonds);
        std::memcpy(m_genotype, genotype_rotbonds, myligand->num_of_rotbonds * sizeof(double));
    }
    
private:
    double m_rotbonds_moving_vectors[g_maxNumRotbonds][3];
    double m_rotbonds_unit_vectors[g_maxNumRotbonds][3];
    
    double m_genotype[g_maxNumRotbonds] = {};
    
    GtKernelCoro kernelMain() override {
        while (true) {
            auto data = co_await m_inputStream->read();
            
            // Process at most m_numRotbondsPerStage bonds and mark them as processed
            for (uint32_t bondCtr = 0; (bondCtr < data.m_numRotbondsPerStage) && data.m_rotbondMask; ++bondCtr) {
                const auto rotbond_id = std::countr_zero(data.m_rotbondMask);
                data.m_rotbondMask &= ~(decltype(data.m_rotbondMask)(1) << rotbond_id);
                
                rotate(&data.m_atomdata.m_atom_idxyzq[1], m_rotbonds_moving_vectors[rotbond_id], m_rotbonds_unit_vectors[rotbond_id], &m_genotype[rotbond_id], 0);
            }
            
            co_await m_outputStream->write(data);
        }
    }

};

/**
 * This performs the final positioning of the conformed ligand in space.
 * Emits raw idxyzq data.
 */
struct Kernel_ChangeConform_GeneralRotation_GlobalMove : IntraE_ComputeKernelBase<ChangeConform_AtomData, InterE_AtomInput> {
    Kernel_ChangeConform_GeneralRotation_GlobalMove(GtContext *ctx, const double *genotype):
        IntraE_ComputeKernelBase<ChangeConform_AtomData, InterE_AtomInput>(ctx), m_genrot_angle(genotype[5])
    {
        double phi = genotype[3] / 180 * M_PI;
        double theta = genotype[4] / 180 * M_PI;

        m_genrot_unitvec [0] = sin(theta)*cos(phi);
        m_genrot_unitvec [1] = sin(theta)*sin(phi);
        m_genrot_unitvec [2] = cos(theta);
        
        std::memcpy(m_globalmove_xyz, genotype, sizeof(m_globalmove_xyz));
    }
    
private:
    double m_genrot_unitvec[3] = {};
    double m_genrot_angle;
    double m_globalmove_xyz[3] = {};
    
    GtKernelCoro kernelMain() override {
        while (true) {
            auto data = co_await m_inputStream->read();
            double *atom_xyz = &data.m_atomdata.m_atom_idxyzq[1];
            BOOST_ASSERT(!data.m_rotbondMask);
            
            const double genrot_movvec[3] = {0, 0, 0};
            rotate(atom_xyz, genrot_movvec, m_genrot_unitvec, &m_genrot_angle, 0);
            
            vec3_accum(atom_xyz, m_globalmove_xyz);
            
            co_await m_outputStream->write(data.m_atomdata);
        }
    }
};

/**
 * Prepares the atom and rotbond data for sending it through the chain of rotate kernels.
 * Also moves the ligand to the origin.
 */
struct Kernel_ChangeConform_BuildRotateInputData : GtKernelBase {
    Kernel_ChangeConform_BuildRotateInputData(GtContext *ctx, const Liganddata *myligand, uint32_t numRotateStages):
        GtKernelBase(ctx), m_numRotbonds(myligand->num_of_rotbonds), m_numRotateStages(numRotateStages)
    {
        // In the AIE version, this should be pre-computed once when loading the ligand.
        get_movvec_to_origo(myligand, m_initial_move_xyz);
    }
    
    auto * atomDataInput() { return m_atomDataInputStream; }
    auto * rotbondsInput() { return m_atomRotbondsInputStream; }
    
    auto * output() { return m_outputStream; }
    
private:
    double m_initial_move_xyz[3];
    uint32_t m_numRotbonds;
    uint32_t m_numRotateStages;

    GtKernelIoStream<InterE_AtomInput> *m_atomDataInputStream = addIoStream<InterE_AtomInput>(g_fifoDepthFor<InterE_AtomInput>);
    GtKernelIoStream<char> *m_atomRotbondsInputStream = addIoStream<char>(g_fifoDepthFor<char>);
    GtKernelIoStream<ChangeConform_AtomData> *m_outputStream = addIoStream<ChangeConform_AtomData>(g_fifoDepthFor<ChangeConform_AtomData>);

    using BondMask = decltype(ChangeConform_AtomData::m_rotbondMask);

    GtKernelCoro kernelMain() override {
        while (true) {
            ChangeConform_AtomData result{.m_atomdata = co_await m_atomDataInputStream->read()};

            BondMask mask = 0;

            for (uint32_t i = 0; i < g_maxNumRotbonds; ++i) {
                const auto isAffectedByRotbond = co_await m_atomRotbondsInputStream->read();
                
                if ((i < m_numRotbonds) && isAffectedByRotbond) {
                    mask |= BondMask(1) << i;
                }
            }
            
            result.m_rotbondMask = mask;
            
            const uint32_t numAffectingRotbonds = std::popcount(mask);
            result.m_numRotbondsPerStage = (numAffectingRotbonds + m_numRotateStages - 1) / m_numRotateStages;
            
            vec3_accum(&result.m_atomdata.m_atom_idxyzq[1], m_initial_move_xyz);
            
            co_await m_outputStream->write(result);
        }
    }
};


static auto change_conform_graphtoy(const Liganddata* myligand, const double genotype []) {
    GtContext ctx{};
    
    const auto numAtoms = size_t(myligand->num_of_atoms);
    
    std::span<const InterE_RawAtomInput> inputAtoms{&myligand->atom_idxyzq[0], numAtoms};
    
    static_assert(sizeof(myligand->atom_rotbonds[0]) == g_maxNumRotbonds);
    std::span<const char> inputRotbonds{&myligand->atom_rotbonds[0][0], g_maxNumRotbonds * numAtoms};
    
    auto& atomInputStream    = ctx.addKernel<GtMemStreamSource<InterE_RawAtomInput, InterE_AtomInput>>(inputAtoms);
    auto& rotbondInputStream = ctx.addKernel<GtMemStreamSource<char>>(inputRotbonds);
    
    std::vector<Kernel_ChangeConform_RotatePartial *> rotateKernels{};
    for (size_t i = 0; i < g_numChangeConformRotateKernels; ++i) {
        auto *kern = rotateKernels.emplace_back(&ctx.addKernel<Kernel_ChangeConform_RotatePartial>(myligand, /* bond rotation angles = */ genotype + 6));

        if (i > 0)
            ctx.connect(rotateKernels.at(i - 1)->output(), kern->input());
    }
    
    BOOST_ASSERT(!rotateKernels.empty());
    
    auto& buildInputData = ctx.addKernel<Kernel_ChangeConform_BuildRotateInputData>(myligand, uint32_t(rotateKernels.size()));
    auto& globalRotation = ctx.addKernel<Kernel_ChangeConform_GeneralRotation_GlobalMove>(genotype);
    
    auto& atomDataSink = ctx.addKernel<GtMemStreamSink<InterE_AtomInput>>();
    
    ctx.connect(atomInputStream.output(), buildInputData.atomDataInput());
    ctx.connect(rotbondInputStream.output(), buildInputData.rotbondsInput());
    ctx.connect(buildInputData.output(), rotateKernels.front()->input());
    ctx.connect(rotateKernels.back()->output(), globalRotation.input());
    ctx.connect(globalRotation.output(), atomDataSink.input());
    
    ctx.runToCompletion();
    
    return std::move(atomDataSink.data());
}

void change_conform(Liganddata* myligand, const double genotype [], int debug) {
    std::unique_ptr<Liganddata> originalLigand = nullptr;
    if (g_graphdumpsEnabled) {
        originalLigand = std::make_unique<Liganddata>(*myligand);
    }

    const auto graphResult = change_conform_graphtoy(myligand, genotype);
    change_conform_original(myligand, genotype, debug);
    
    static_assert(sizeof(InterE_AtomInput) == sizeof(InterE_RawAtomInput));
#ifndef __clang__
    static_assert(std::is_pointer_interconvertible_with_class(&InterE_AtomInput::m_atom_idxyzq));
#endif
    
    BOOST_ASSERT(graphResult.size() == size_t(myligand->num_of_atoms));
    const size_t numBytes = graphResult.size() * sizeof(graphResult[0]);
    BOOST_ASSERT(std::memcmp(graphResult.data(), &myligand->atom_idxyzq[0], numBytes) == 0);

    if (g_graphdumpsEnabled) {
        static uint32_t s_dumpIndex = 0;
        const std::string baseName = g_dumpDirectory + "/change_conform_" + std::to_string(s_dumpIndex++);

        dumpLigand((baseName + "_ligand_original.bin").data(), originalLigand.get());
        dumpLigand((baseName + "_ligand_rotated.bin").data(), myligand);
        dumpStructRaw((baseName + "_genotype.bin").data(), (const char *)genotype, (6 + myligand->num_of_rotbonds) * sizeof(double));
    }
}

extern "C" void enable_graphdumps() {
    g_graphdumpsEnabled = true;
}
