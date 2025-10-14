#include <cstddef>
#include <cstdint>
#include <array>
#include <type_traits>
#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <cstring>
#include <utility>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <bit>
#include <unordered_map>

#include <boost/assert.hpp>

#include "fdock_intraE_luts.hpp"
#include "miscellaneous_inline.h"
#include "processligand_inline.h"
#include "processligand.h"
#include "cgsim/cgsim.hpp"
#include "json.hpp"

#ifdef HAVE_TAPASCO
#include <tapasco.hpp>

static constexpr const char s_tpc_vlnv_intraE[] = "cgsim_autogen:cgsim_graphs:intraE_graph_hls:1.0";
static constexpr const char s_tpc_vlnv_interE[] = "cgsim_autogen:cgsim_graphs:interE_graph_hls:1.0";
static constexpr const char s_tpc_vlnv_changeConform[] = "cgsim_autogen:cgsim_graphs:changeConform_graph_hls:1.0";

static tapasco::Tapasco& get_tapasco() {
    static tapasco::Tapasco tpc{};
    return tpc;
}

static tapasco::PEId get_tapasco_pe(const char *vlnv) {
    static std::unordered_map<std::string_view, tapasco::PEId> known_pes{};

    if (auto iter = known_pes.find(vlnv); iter != known_pes.end()) {
        return iter->second;
    }

    auto& tpc = get_tapasco();
    auto peid = tpc.get_pe_id(vlnv);
    known_pes[vlnv] = peid;

    std::cerr << "\nResolved TaPaSCo PEID: " << vlnv << " -> " << peid << std::endl;

    if (!tpc.kernel_pe_count(peid)) {
        throw std::runtime_error("Need at least 1 PE instance");
    }

    return peid;
}

template<typename T>
static auto tapasco_inbuf(std::span<const T> buf) {
    return tapasco::makeInOnly(tapasco::makeWrappedPointer(buf.data(), buf.size_bytes()));
}

template<typename T>
static auto tapasco_outbuf(std::span<T> buf) {
    return tapasco::makeOutOnly(tapasco::makeWrappedPointer(buf.data(), buf.size_bytes()));
}

#endif

using namespace ttlhacker::cgsim;

static constexpr auto s_iop_rtp = IoPortEndpointOptions{.m_isSingleWrite = true};

static constexpr auto g_maxNumAtoms = 256;
static constexpr auto g_maxNumAtomTypes = 14;
static constexpr auto g_numDistanceIDs = 2048;
static constexpr auto g_outOfGridPenalty = 1 << 24;
static constexpr auto g_peratomOutOfGridPenalty = 100000;
static constexpr auto g_maxNumRotbonds = 32;

// The AIE's AXI4 FIFOs have a depth of four 32-bit values (16 bytes total).
static constexpr size_t g_aieAxiFifoDepthBytes = 16;

// FIFO depth that better emulates the AI Engines.
template<typename T>
static constexpr size_t g_fifoDepthFor = std::max(size_t(1), g_aieAxiFifoDepthBytes / sizeof(T));


static bool g_graphdumpsEnabled = false;
static const std::string g_dumpDirectory = "dumps";

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
#pragma HLS unroll off
        for (uint32_t y = 0; y < num_atoms; ++y) {
#pragma HLS unroll off
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
#pragma HLS bind_storage variable=atom_idxyzq type=ram_2p impl=bram

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
#pragma HLS allocation operation instances=dmul limit=2

    const double dcutoff = co_await dcutoff_in.get();

    // Make a local copy of the H-bond LUT
    constexpr size_t num_hbond_lut_entries = g_maxNumAtomTypes * g_maxNumAtomTypes;
    char is_hbond_lut[num_hbond_lut_entries];

    for (size_t i = 0; i < num_hbond_lut_entries; ++i) {
#pragma HLS unroll off
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
#pragma HLS allocation operation instances=dmul limit=1
#pragma HLS allocation operation instances=dadd limit=1

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

/**
 * Multiplies m_vdW1 and m_vdW2 with the appropriate power of the distance.
 */
COMPUTE_KERNEL(hls, kernel_IntraE_ScaleVWparsWithDistance,
    KernelReadPort<IntraE_AtomPair> atom_data_in,
    KernelWritePort<IntraE_AtomPair> atom_data_out
) {
#pragma HLS allocation operation instances=dmul limit=1

    while (true) {
        auto data = co_await atom_data_in.get();

        if (!data.m_terminate_processing) {
            const auto did = data.m_distanceID;

            data.m_vdW1 *= fdock_luts::intra_r_12_table[did];
            data.m_vdW2 *= (data.m_isHBond ? fdock_luts::intra_r_10_table[did] : fdock_luts::intra_r_6_table[did]);
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
    KernelMemoryPort<double> outs
) {
#pragma HLS allocation operation instances=dmul limit=3

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

        el +=
            q1 * q2
            * (epsrScale / fdock_luts::intra_r_epsr_table_unscaled[data.m_distanceID]);

        desolv +=
            (data.m_s1 * data.m_v2 + data.m_s2 * data.m_v1)
            * (desolvScale * fdock_luts::intra_desolv_table_unscaled[data.m_distanceID]);
    }

    //co_await vW_out.put(vW);
    //co_await el_out.put(el);
    //co_await desolv_out.put(desolv);
    outs[0] = vW;
    outs[1] = el;
    outs[2] = desolv;
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
    IoConnector<double> AD4_coeff_desolv_in,
    IoConnector<double> out_buf
) {
    IoConnector<AtomIndexPair> atom_pairs;
    IoConnector<IntraE_AtomPair>
        atom_data_fetched,
        atom_data_distance_checked,
        atom_data_with_volume,
        atom_data_with_vw_fetched,
        atom_data_with_vw_scaled;

    CGSIM_AUTO_NAME(num_atoms_in);
    CGSIM_AUTO_NAME(intraE_contributors_buf);
    CGSIM_AUTO_NAME(atom_idxyzq_buf);
    CGSIM_AUTO_NAME(is_hbond_lut_buf);
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
    CGSIM_AUTO_NAME(out_buf);

    kernel_IntraE_GenAtomPairIndices(num_atoms_in, intraE_contributors_buf, atom_pairs);
    kernel_IntraE_FetchAtomData(num_atoms_in, atom_pairs, atom_idxyzq_buf, atom_data_fetched);
    kernel_IntraE_SetDistanceID_CheckHBond(atom_data_fetched, atom_data_distance_checked, dcutoff_in, is_hbond_lut_buf);
    kernel_IntraE_Volume_Solpar(atom_data_distance_checked, atom_data_with_volume, qasp_in, volume_buf, solpar_buf);
    kernel_IntraE_FetchVWpars(atom_data_with_volume, atom_data_with_vw_fetched, vwpars_a_buf, vwpars_b_buf, vwpars_c_buf, vwpars_d_buf);
    kernel_IntraE_ScaleVWparsWithDistance(atom_data_with_vw_fetched, atom_data_with_vw_scaled);
    kernel_IntraE_Compute_VW_EL_Desolv(atom_data_with_vw_scaled, scaled_AD4_coeff_elec_in, AD4_coeff_desolv_in, out_buf);

    return std::tuple();
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

    std::vector<double> out_buf(3, 0.0); // vW, el, desolv
    double& vW = out_buf[0];
    double& el = out_buf[1];
    double& desolv = out_buf[2];

#ifdef HAVE_TAPASCO
    auto& tpc = get_tapasco();
    const auto peid = get_tapasco_pe(s_tpc_vlnv_intraE);

    auto intraE_contributors_tpc = tapasco_inbuf<char>(intraE_contributors_buf);
    auto atom_idxyzq_tpc = tapasco_inbuf<double>(atom_idxyzq_buf);
    auto is_hbond_lut_tpc = tapasco_inbuf<char>(is_hbond_lut_buf);
    auto volume_tpc = tapasco_inbuf<double>(std::span<const double>(myligand->volume, g_maxNumAtomTypes));
    auto solpar_tpc = tapasco_inbuf<double>(std::span<const double>(myligand->solpar, g_maxNumAtomTypes));
    auto vwpars_a_tpc = tapasco_inbuf<double>(vwpars_a_buf);
    auto vwpars_b_tpc = tapasco_inbuf<double>(vwpars_b_buf);
    auto vwpars_c_tpc = tapasco_inbuf<double>(vwpars_c_buf);
    auto vwpars_d_tpc = tapasco_inbuf<double>(vwpars_d_buf);
    auto out_tpc = tapasco_outbuf<double>(out_buf);

    auto job = tpc.launch(peid,
        myligand->num_of_atoms,
        intraE_contributors_tpc,
        atom_idxyzq_tpc,
        is_hbond_lut_tpc,
        volume_tpc,
        solpar_tpc,
        vwpars_a_tpc,
        vwpars_b_tpc,
        vwpars_c_tpc,
        vwpars_d_tpc,
        dcutoff,
        qasp,
        scaled_AD4_coeff_elec,
        AD4_coeff_desolv,
        out_tpc
    );

    job();
#else
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
        memBuffer(out_buf)
    );

    // Warn about deadlocks
    result.dump(std::cerr);
#endif

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
    bool m_terminate_processing = false;
    bool m_isOutOfGrid = false;

    int m_type_id  = 0;

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
    bool m_isOutOfGrid = false;
    bool m_terminate_processing = false;

    double m_atomTypeGridEnergy = 0;
    double m_electrostaticGridEnergy = 0;
    double m_desolvationGridEnergy = 0;
};


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

COMPUTE_KERNEL(hls, kernel_interE_BuildAtomData,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelReadPort<double, s_iop_rtp> outofgrid_tolerance_in,
    KernelReadPort<int, s_iop_rtp> grid_size_x_in,
    KernelReadPort<int, s_iop_rtp> grid_size_y_in,
    KernelReadPort<int, s_iop_rtp> grid_size_z_in,
    KernelMemoryPort<const double> atom_idxyzq_buf,
    KernelWritePort<InterE_AtomData> atom_data_out
) {
#pragma HLS allocation operation instances=dmul limit=3

    uint32_t num_atoms = co_await num_atoms_in.get();
    num_atoms = std::min(num_atoms, uint32_t(g_maxNumAtoms));

    const double outofgrid_tolerance = co_await outofgrid_tolerance_in.get();
    const int grid_size_x = co_await grid_size_x_in.get();
    const int grid_size_y = co_await grid_size_y_in.get();
    const int grid_size_z = co_await grid_size_z_in.get();

    const std::array<int, 3> size_xyz = {grid_size_x, grid_size_y, grid_size_z};

    for (uint32_t i = 0; i < num_atoms; ++i) {
        InterE_AtomData outputData{};

        double inputData[5];
        for (size_t j = 0; j < 5; ++j) {
            inputData[j] = atom_idxyzq_buf[i * 5 + j];
        }

        outputData.m_type_id = int(inputData[0]);

        double x = inputData[1];
        double y = inputData[2];
        double z = inputData[3];

        outputData.m_isOutOfGrid = interE_nudgeGridCoordsIntoBounds(x, y, z, outofgrid_tolerance, size_xyz);

        if (!outputData.m_isOutOfGrid) {
            outputData.m_q = inputData[4];

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

        co_await atom_data_out.put(outputData);
    }

    // Terminate pipeline
    co_await atom_data_out.put(InterE_AtomData{.m_terminate_processing = true});
}

static constexpr uint32_t s_terminate_dram_reader_sentinel = UINT32_MAX;

COMPUTE_KERNEL(hls, kernel_interE_GenerateDramAddresses,
    KernelReadPort<InterE_AtomData> atom_data_in,
    KernelWritePort<uint32_t> address_out,
    KernelReadPort<int, s_iop_rtp> grid_size_x_in,
    KernelReadPort<int, s_iop_rtp> grid_size_y_in,
    KernelReadPort<int, s_iop_rtp> grid_size_z_in,
    KernelReadPort<int, s_iop_rtp> num_of_atypes_in
) {
    const int grid_size_x = co_await grid_size_x_in.get();
    const int grid_size_y = co_await grid_size_y_in.get();
    const int grid_size_z = co_await grid_size_z_in.get();
    const int num_of_atypes = co_await num_of_atypes_in.get();

    const std::array<int, 3> size_xyz = {grid_size_x, grid_size_y, grid_size_z};

    while (true) {
        const auto data = co_await atom_data_in.get();

        if (data.m_terminate_processing) {
            break;
        }

        if (data.m_isOutOfGrid)
            continue;

        std::array<int, 8> coordOffsets = {
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_low, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_low, data.m_x_high),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_high, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_high, data.m_x_high),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_low, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_low, data.m_x_high),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_high, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_high, data.m_x_high)
        };

        std::array<int, 3> gridNumbers = {
            data.m_type_id,     // energy contribution of the current grid type
            num_of_atypes,      // energy contribution of the electrostatic grid
            num_of_atypes + 1   // energy contribution of the desolvation grid
        };

        for (const int gridNumber: gridNumbers) {
            const int gridOffset = interE_gridNumberToArrayOffset(size_xyz, gridNumber);
            for (const int coordOffset: coordOffsets) {
                const int finalAddr = gridOffset + coordOffset;
                //BOOST_ASSERT(finalAddr >= 0);
                co_await address_out.put(uint32_t(finalAddr));
            }
        }
    }

    // Terminate DRAM reader
    co_await address_out.put(s_terminate_dram_reader_sentinel);
}

COMPUTE_KERNEL(hls, kernel_interE_InterpolateEnergy,
    KernelReadPort<InterE_AtomData> atom_data_in,
    KernelReadPort<double> dram_data_in,
    KernelWritePort<InterE_AtomEnergy> atom_energy_out
) {
#pragma HLS allocation operation instances=dmul limit=4

    enum class GridType { ATOM, ELECTROSTATIC, DESOLVATION };

    while (true) {
        const auto data = co_await atom_data_in.get();

        if (data.m_terminate_processing) {
            co_await atom_energy_out.put(InterE_AtomEnergy{.m_terminate_processing = true});
            break;
        }

        InterE_AtomEnergy result{.m_isOutOfGrid = data.m_isOutOfGrid};

        if (!data.m_isOutOfGrid) {
            for (const GridType t: {GridType::ATOM, GridType::ELECTROSTATIC, GridType::DESOLVATION}) {
                double cube[2][2][2];

                cube [0][0][0] = co_await dram_data_in.get();
                cube [1][0][0] = co_await dram_data_in.get();
                cube [0][1][0] = co_await dram_data_in.get();
                cube [1][1][0] = co_await dram_data_in.get();
                cube [0][0][1] = co_await dram_data_in.get();
                cube [1][0][1] = co_await dram_data_in.get();
                cube [0][1][1] = co_await dram_data_in.get();
                cube [1][1][1] = co_await dram_data_in.get();

                const double interpolated = trilin_interpol_inline(cube, data.m_weights);

                switch (t) {
                    case GridType::ATOM:            result.m_atomTypeGridEnergy         = interpolated;                     break;
                    case GridType::ELECTROSTATIC:   result.m_electrostaticGridEnergy    = data.m_q * interpolated;          break;
                    case GridType::DESOLVATION:     result.m_desolvationGridEnergy      = fabs(data.m_q) * interpolated;    break;
                }
            }
        }

        co_await atom_energy_out.put(result);
    }
}

COMPUTE_KERNEL_TEMPLATE(hls, kernel_fdock_ReadDram,
    KernelReadPort<uint32_t> address_in,
    KernelMemoryPort<const T> dram_buf,
    KernelWritePort<T> data_out
) {
    while (true) {
        const uint32_t addr = co_await address_in.get();

        if (addr == s_terminate_dram_reader_sentinel) {
            break;
        }

        const T data = dram_buf[addr];
        co_await data_out.put(data);
    }
}

COMPUTE_KERNEL(hls, kernel_interE_AccumulateResults,
    KernelReadPort<InterE_AtomEnergy> atom_energy_in,
    KernelReadPort<bool, s_iop_rtp> enable_peratom_outputs_in,
    KernelMemoryPort<double> vdw_buf,
    KernelMemoryPort<double> elec_buf,
    KernelMemoryPort<double> scalar_out_buf
) {
#pragma HLS allocation operation instances=dadd limit=2

    const bool enable_peratom_outputs = co_await enable_peratom_outputs_in.get();

    double interE = 0;
    double elecE  = 0;

    uint32_t peratom_index = 0;

    while (true) {
        const auto data = co_await atom_energy_in.get();

        if (data.m_terminate_processing) {
            break;
        }

        auto vdW  = data.m_atomTypeGridEnergy;
        auto elec = data.m_electrostaticGridEnergy;

        interE += vdW;
        interE += elec;
        interE += data.m_desolvationGridEnergy;

        elecE += elec;

        if (data.m_isOutOfGrid) {
            interE += g_outOfGridPenalty;
            elec = vdW = g_peratomOutOfGridPenalty;
        }

        if (enable_peratom_outputs) {
            //BOOST_ASSERT(peratom_index < g_maxNumAtoms);
            vdw_buf[peratom_index] = vdW;
            elec_buf[peratom_index] = elec;
            peratom_index++;
        }
    }

    //co_await interE_out.put(interE);
    //co_await elecE_out.put(elecE);

    scalar_out_buf[0] = interE;
    scalar_out_buf[1] = elecE;
}

COMPUTE_GRAPH constexpr auto interE_graph = make_compute_graph_v<[] (
    IoConnector<uint32_t> num_atoms_in,
    IoConnector<double> outofgrid_tolerance_in,
    IoConnector<int> grid_size_x_in,
    IoConnector<int> grid_size_y_in,
    IoConnector<int> grid_size_z_in,
    IoConnector<int> num_of_atypes_in,
    IoConnector<bool> enable_peratom_outputs_in,
    IoConnector<const double> atom_idxyzq_buf,
    IoConnector<const double> grid_buf,
    IoConnector<double> peratom_vdw_buf,
    IoConnector<double> peratom_elec_buf,
    IoConnector<double> scalar_out_buf
) {
    IoConnector<InterE_AtomData> atom_data_stream;
    IoConnector<uint32_t> dram_address_stream;
    IoConnector<double> dram_data_stream;
    IoConnector<InterE_AtomEnergy> atom_energy_stream;

    CGSIM_AUTO_NAME(num_atoms_in);
    CGSIM_AUTO_NAME(outofgrid_tolerance_in);
    CGSIM_AUTO_NAME(grid_size_x_in);
    CGSIM_AUTO_NAME(grid_size_y_in);
    CGSIM_AUTO_NAME(grid_size_z_in);
    CGSIM_AUTO_NAME(num_of_atypes_in);
    CGSIM_AUTO_NAME(enable_peratom_outputs_in);
    CGSIM_AUTO_NAME(atom_idxyzq_buf);
    CGSIM_AUTO_NAME(grid_buf);
    CGSIM_AUTO_NAME(peratom_vdw_buf);
    CGSIM_AUTO_NAME(peratom_elec_buf);
    CGSIM_AUTO_NAME(scalar_out_buf);

    kernel_interE_BuildAtomData(
        num_atoms_in,
        outofgrid_tolerance_in,
        grid_size_x_in,
        grid_size_y_in,
        grid_size_z_in,
        atom_idxyzq_buf,
        atom_data_stream
    );

    kernel_interE_GenerateDramAddresses(
        atom_data_stream,
        dram_address_stream,
        grid_size_x_in,
        grid_size_y_in,
        grid_size_z_in,
        num_of_atypes_in
    );

    kernel_fdock_ReadDram<double>(
        dram_address_stream,
        grid_buf,
        dram_data_stream
    );

    kernel_interE_InterpolateEnergy(
        atom_data_stream,
        dram_data_stream,
        atom_energy_stream
    );

    kernel_interE_AccumulateResults(
        atom_energy_stream,
        enable_peratom_outputs_in,
        peratom_vdw_buf,
        peratom_elec_buf,
        scalar_out_buf
    );

    return std::tuple();
}>;


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
    // Copy 2D buffers into local (flat) ones
    static_assert(sizeof(myligand->atom_idxyzq) == (g_maxNumAtoms * 5 * sizeof(double)));
    std::vector<double> atom_idxyzq_buf(g_maxNumAtoms * 5);
    std::memcpy(atom_idxyzq_buf.data(), myligand->atom_idxyzq, sizeof(myligand->atom_idxyzq));

    const auto numGridMemElems = getNumGridElems(mygrid);
    std::span<const double> gridMemoryRegion{fgrids, size_t(numGridMemElems)};

    // Output buffers
    std::vector<double> peratom_vdw_buf{1, 0.0};
    std::vector<double> peratom_elec_buf{1, 0.0};
    if (enablePerAtomOutputs) {
        peratom_vdw_buf.resize(myligand->num_of_atoms);
        peratom_elec_buf.resize(myligand->num_of_atoms);
    }

    // Run graph
    std::vector<double> out_buf(2, 0.0); // interE, elecE
    double& interE = out_buf[0];
    double& elecE  = out_buf[1];

#ifdef HAVE_TAPASCO
    auto& tpc = get_tapasco();
    const auto peid = get_tapasco_pe(s_tpc_vlnv_interE);

    auto atom_idxyzq_tpc = tapasco_inbuf<double>(atom_idxyzq_buf);
    auto grid_tpc = tapasco_inbuf<double>(gridMemoryRegion);
    auto peratom_vdw_tpc = tapasco_outbuf<double>(peratom_vdw_buf);
    auto peratom_elec_tpc = tapasco_outbuf<double>(peratom_elec_buf);
    auto out_tpc = tapasco_outbuf<double>(out_buf);

    auto job = tpc.launch(peid,
        myligand->num_of_atoms,
        outofgrid_tolerance,
        mygrid->size_xyz[0],
        mygrid->size_xyz[1],
        mygrid->size_xyz[2],
        myligand->num_of_atypes,
        uint32_t(enablePerAtomOutputs),
        atom_idxyzq_tpc,
        grid_tpc,
        peratom_vdw_tpc,
        peratom_elec_tpc,
        out_tpc
    );

    job();
#else
    const auto result = interE_graph(
        ScalarDataSource<uint32_t>(myligand->num_of_atoms),
        ScalarDataSource<double>(outofgrid_tolerance),
        ScalarDataSource<int>(mygrid->size_xyz[0]),
        ScalarDataSource<int>(mygrid->size_xyz[1]),
        ScalarDataSource<int>(mygrid->size_xyz[2]),
        ScalarDataSource<int>(myligand->num_of_atypes),
        ScalarDataSource<bool>(enablePerAtomOutputs),
        memBuffer(atom_idxyzq_buf),
        RuntimeMemoryBuffer(gridMemoryRegion),
        memBuffer(peratom_vdw_buf),
        memBuffer(peratom_elec_buf),
        memBuffer(out_buf)
    );

    result.dump(std::cerr);
#endif

    return {
        .m_interE = interE,
        .m_elecE = elecE,
        .m_peratomVdW = std::move(peratom_vdw_buf),
        .m_peratomElec = std::move(peratom_elec_buf)
    };
}

double calc_interE(const Gridinfo* mygrid, const Liganddata* myligand, const double* fgrids, double outofgrid_tolerance, int debug) {
    const double originalResult = calc_interE_original(mygrid, myligand, fgrids, outofgrid_tolerance, debug);
    const double graphResult = calc_interE_graphtoy(mygrid, myligand, fgrids, outofgrid_tolerance, false).m_interE;

    if (graphResult != originalResult) {
        std::cerr << "InterE mismatch: original=" << originalResult << " graph=" << graphResult << "\n";
    }

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

    if (graphResult.m_peratomVdW.size() != num_atoms || graphResult.m_peratomElec.size() != num_atoms) {
        std::cerr << "InterE per-atom output size mismatch: expected " << num_atoms
                  << ", got " << graphResult.m_peratomVdW.size() << " vdw and "
                  << graphResult.m_peratomElec.size() << " elec\n";
        return;
    }

    if (!std::ranges::equal(graphResult.m_peratomVdW, std::span(peratom_vdw, num_atoms))) {
        std::cerr << "InterE per-atom VdW mismatch\n";
    }

    if (!std::ranges::equal(graphResult.m_peratomElec, std::span(peratom_elec, num_atoms))) {
        std::cerr << "InterE per-atom Elec mismatch\n";
    }
}


struct ChangeConform_AtomData {
    InterE_AtomInput m_atomdata;
    uint32_t m_rotbondMask;

    bool m_terminate_processing = false;

    //static_assert(sizeof(m_rotbondMask) * CHAR_BIT >= g_maxNumRotbonds);
};

static void vec3_accum(double *vec_a, const double *vec_b) {
    for (size_t i = 0; i < 3; ++i) {
        vec_a[i] += vec_b[i];
    }
}

COMPUTE_KERNEL(hls, kernel_ChangeConform_Rotate,
    KernelReadPort<ChangeConform_AtomData> atom_data_in,
    KernelReadPort<uint32_t, s_iop_rtp> num_rotbonds_in,
    KernelReadPort<double> genotype_sincos_in,
    KernelMemoryPort<const double> rotbonds_moving_vectors_buf,
    KernelMemoryPort<const double> rotbonds_unit_vectors_buf,
    KernelMemoryPort<double> output_buf
) {
#pragma HLS allocation operation instances=dmul limit=2
//#pragma HLS allocation function instances=rotate_precomputed_sincos limit=1

    double genrot_unitvec[3];
    double globalmove_xyz[3];

    double rotbonds_moving_vectors[g_maxNumRotbonds][3];
    double rotbonds_unit_vectors[g_maxNumRotbonds][3];
    double genotype_sincos[g_maxNumRotbonds][2];

    const uint32_t num_rotbonds = co_await num_rotbonds_in.get();

    // Grab the rotational bond data from memory and cache it locally
    for (size_t i = 0; i < num_rotbonds; ++i) {
#pragma HLS unroll off

        for (size_t j = 0; j < 3; ++j) {
            rotbonds_moving_vectors[i][j] = rotbonds_moving_vectors_buf[i * 3 + j];
            rotbonds_unit_vectors[i][j] = rotbonds_unit_vectors_buf[i * 3 + j];
        }
    }

    // Read global move and general rotation values
    for (uint32_t i = 0; i < 3; ++i) {
        globalmove_xyz[i] = co_await genotype_sincos_in.get();
    }

    const double sin_phi = co_await genotype_sincos_in.get();
    const double cos_phi = co_await genotype_sincos_in.get();
    const double sin_theta = co_await genotype_sincos_in.get();
    const double cos_theta = co_await genotype_sincos_in.get();

    genrot_unitvec[0] = sin_theta * cos_phi;
    genrot_unitvec[1] = sin_theta * sin_phi;
    genrot_unitvec[2] = cos_theta;

    double genrot_angle_sincos[2];
    for (uint32_t i = 0; i < 2; ++i) {
        genrot_angle_sincos[i] = co_await genotype_sincos_in.get();
    }

    // Read genotype sin/cos values for rotbonds
    for (size_t i = 0; i < num_rotbonds; ++i) {
#pragma HLS unroll off

        for (size_t j = 0; j < 2; ++j) {
            genotype_sincos[i][j] = co_await genotype_sincos_in.get();
        }
    }

    uint32_t atom_idx = 0;

    while (true) {
        auto data = co_await atom_data_in.get();

        if (data.m_terminate_processing) break;

        double *atom_xyz = &data.m_atomdata.m_atom_idxyzq[1];

        // Process all rotbonds at once
        for (uint32_t bondCtr = 0; bondCtr < g_maxNumRotbonds; ++bondCtr) {
#pragma HLS unroll off

            if (data.m_rotbondMask & (decltype(data.m_rotbondMask)(1) << bondCtr)) {
                rotate_precomputed_sincos(
                    atom_xyz,
                    rotbonds_moving_vectors[bondCtr],
                    rotbonds_unit_vectors[bondCtr],
                    +genotype_sincos[bondCtr]
                );
            }
        }

        // Apply general rotation
        const double genrot_movvec[3] = {0, 0, 0};
        rotate_precomputed_sincos(atom_xyz, genrot_movvec, genrot_unitvec, +genrot_angle_sincos);

        // Apply global move
        vec3_accum(atom_xyz, globalmove_xyz);

        // Write output
        for (uint32_t i = 0; i < 5; ++i) {
            output_buf[atom_idx * 5 + i] = data.m_atomdata.m_atom_idxyzq[i];
        }

        atom_idx++;
    }
}

/**
 * Prepares the atom and rotbond data for sending it through the chain of rotate kernels.
 * Also moves the ligand to the origin.
 */
COMPUTE_KERNEL(hls, kernel_ChangeConform_BuildRotateInputData,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelReadPort<uint32_t, s_iop_rtp> num_rotbonds_in,
    KernelReadPort<double, s_iop_rtp> initial_move_x_in,
    KernelReadPort<double, s_iop_rtp> initial_move_y_in,
    KernelReadPort<double, s_iop_rtp> initial_move_z_in,
    KernelMemoryPort<const double> atom_idxyzq_buf,
    KernelMemoryPort<const char> atom_rotbonds_buf,
    KernelWritePort<ChangeConform_AtomData> atoms_out
) {
    const uint32_t num_atoms = co_await num_atoms_in.get();
    const uint32_t num_rotbonds = co_await num_rotbonds_in.get();

    const double initial_move_xyz[3] = {
        co_await initial_move_x_in.get(),
        co_await initial_move_y_in.get(),
        co_await initial_move_z_in.get()
    };

    using BondMask = decltype(ChangeConform_AtomData::m_rotbondMask);

    for (uint32_t atomIdx = 0; atomIdx < num_atoms; ++atomIdx) {
        ChangeConform_AtomData result{};

        for (size_t i = 0; i < 5; ++i) {
            result.m_atomdata.m_atom_idxyzq[i] = atom_idxyzq_buf[atomIdx * 5 + i];
        }

        BondMask mask = 0;

        for (uint32_t rotbondIdx = 0; rotbondIdx < num_rotbonds; ++rotbondIdx) {
            const char isAffectedByRotbond = atom_rotbonds_buf[atomIdx * g_maxNumRotbonds + rotbondIdx];

            if (isAffectedByRotbond) {
                mask |= BondMask(1) << rotbondIdx;
            }
        }

        result.m_rotbondMask = mask;

        vec3_accum(&result.m_atomdata.m_atom_idxyzq[1], initial_move_xyz);

        co_await atoms_out.put(result);
    }

    // Terminate pipeline
    co_await atoms_out.put(ChangeConform_AtomData{.m_terminate_processing = true});
}

/**
 * Computes sin(x), cos(x), sin(x/2), cos(x/2) for each value in the genotype buffer.
 */
COMPUTE_KERNEL(hls, kernel_ChangeConform_precomputeTrig,
    KernelReadPort<uint32_t, s_iop_rtp> num_rotbonds_in,
    KernelMemoryPort<const double> genotype_buf,
    KernelWritePort<double> trig_out
) {
#pragma HLS allocation operation instances=dmul limit=2

    const uint32_t genotype_buf_size = 6 + co_await num_rotbonds_in.get();

    for (uint32_t i = 0; i < genotype_buf_size; ++i) {
#pragma HLS unroll off

        const double angle = genotype_buf[i];

        if (i < 3) {
            // No trig needed for global move
            co_await trig_out.put(angle);

        } else if (i < 5) {
            // phi, theta
            co_await trig_out.put(sin(angle / 180 * M_PI));
            co_await trig_out.put(cos(angle / 180 * M_PI));

        } else {
            // Global rotation, rotbonds
            co_await trig_out.put(sin(angle / 2 / 180 * M_PI));
            co_await trig_out.put(cos(angle / 2 / 180 * M_PI));
        }
    }
}

COMPUTE_GRAPH constexpr auto changeConform_graph = make_compute_graph_v<[] (
    IoConnector<uint32_t> num_atoms_in,
    IoConnector<uint32_t> num_rotbonds_in,
    IoConnector<double> initial_move_x_in,
    IoConnector<double> initial_move_y_in,
    IoConnector<double> initial_move_z_in,
    IoConnector<const double> atom_data_buf,
    IoConnector<const char> rotbonds_buf,
    IoConnector<const double> rotbonds_moving_vectors_buf,
    IoConnector<const double> rotbonds_unit_vectors_buf,
    IoConnector<const double> genotype_buf,
    IoConnector<double> output_buf
) {
    IoConnector<ChangeConform_AtomData> atoms_read;
    IoConnector<double> genotype_sincos_precomputed;

    CGSIM_AUTO_NAME(num_atoms_in);
    CGSIM_AUTO_NAME(num_rotbonds_in);
    CGSIM_AUTO_NAME(initial_move_x_in);
    CGSIM_AUTO_NAME(initial_move_y_in);
    CGSIM_AUTO_NAME(initial_move_z_in);
    CGSIM_AUTO_NAME(atom_data_buf);
    CGSIM_AUTO_NAME(rotbonds_buf);
    CGSIM_AUTO_NAME(rotbonds_moving_vectors_buf);
    CGSIM_AUTO_NAME(rotbonds_unit_vectors_buf);
    CGSIM_AUTO_NAME(genotype_buf);
    CGSIM_AUTO_NAME(output_buf);

    kernel_ChangeConform_precomputeTrig(
        num_rotbonds_in,
        genotype_buf,
        genotype_sincos_precomputed
    );

    kernel_ChangeConform_BuildRotateInputData(
        num_atoms_in,
        num_rotbonds_in,
        initial_move_x_in,
        initial_move_y_in,
        initial_move_z_in,
        atom_data_buf,
        rotbonds_buf,
        atoms_read
    );

    kernel_ChangeConform_Rotate(
        atoms_read,
        num_rotbonds_in,
        genotype_sincos_precomputed,
        rotbonds_moving_vectors_buf,
        rotbonds_unit_vectors_buf,
        output_buf
    );

    return std::tuple();
}>;


static auto change_conform_graphtoy(const Liganddata* myligand, const double genotype []) {
    // Copy 2D buffers into local (flat) ones
    static_assert(sizeof(myligand->atom_idxyzq) == (g_maxNumAtoms * 5 * sizeof(double)));
    std::vector<double> atom_idxyzq_buf(g_maxNumAtoms * 5);
    std::memcpy(atom_idxyzq_buf.data(), myligand->atom_idxyzq, sizeof(myligand->atom_idxyzq));

    static_assert(sizeof(myligand->atom_rotbonds) == (g_maxNumAtoms * g_maxNumRotbonds * sizeof(char)));
    std::vector<char> atom_rotbonds_buf(g_maxNumAtoms * g_maxNumRotbonds);
    std::memcpy(atom_rotbonds_buf.data(), myligand->atom_rotbonds, sizeof(myligand->atom_rotbonds));

    static_assert(sizeof(myligand->rotbonds_moving_vectors) == (g_maxNumRotbonds * 3 * sizeof(double)));
    std::vector<double> rotbonds_moving_vectors_buf(g_maxNumRotbonds * 3);
    std::memcpy(rotbonds_moving_vectors_buf.data(), myligand->rotbonds_moving_vectors, sizeof(myligand->rotbonds_moving_vectors));

    static_assert(sizeof(myligand->rotbonds_unit_vectors) == (g_maxNumRotbonds * 3 * sizeof(double)));
    std::vector<double> rotbonds_unit_vectors_buf(g_maxNumRotbonds * 3);
    std::memcpy(rotbonds_unit_vectors_buf.data(), myligand->rotbonds_unit_vectors, sizeof(myligand->rotbonds_unit_vectors));

    // Output buffer
    std::vector<double> output_buf(g_maxNumAtoms * 5);

    // Compute initial move to origo
    double initial_move_xyz[3];
    get_movvec_to_origo(myligand, initial_move_xyz);

    const auto genotype_span = std::span<const double>(genotype, 6 + myligand->num_of_rotbonds);

#ifdef HAVE_TAPASCO
    auto& tpc = get_tapasco();
    const auto peid = get_tapasco_pe(s_tpc_vlnv_changeConform);

    auto atom_idxyzq_tpc = tapasco_inbuf<double>(atom_idxyzq_buf);
    auto atom_rotbonds_tpc = tapasco_inbuf<char>(atom_rotbonds_buf);
    auto rotbonds_moving_vectors_tpc = tapasco_inbuf<double>(rotbonds_moving_vectors_buf);
    auto rotbonds_unit_vectors_tpc = tapasco_inbuf<double>(rotbonds_unit_vectors_buf);
    auto genotype_tpc = tapasco_inbuf<double>(genotype_span);
    auto output_tpc = tapasco_outbuf<double>(output_buf);

    auto job = tpc.launch(peid,
        myligand->num_of_atoms,
        myligand->num_of_rotbonds,
        initial_move_xyz[0],
        initial_move_xyz[1],
        initial_move_xyz[2],
        atom_idxyzq_tpc,
        atom_rotbonds_tpc,
        rotbonds_moving_vectors_tpc,
        rotbonds_unit_vectors_tpc,
        genotype_tpc,
        output_tpc
    );

    job();
#else
    // Run graph
    const auto result = changeConform_graph(
        ScalarDataSource<uint32_t>(myligand->num_of_atoms),
        ScalarDataSource<uint32_t>(myligand->num_of_rotbonds),
        ScalarDataSource<double>(initial_move_xyz[0]),
        ScalarDataSource<double>(initial_move_xyz[1]),
        ScalarDataSource<double>(initial_move_xyz[2]),
        memBuffer(atom_idxyzq_buf),
        memBuffer(atom_rotbonds_buf),
        memBuffer(rotbonds_moving_vectors_buf),
        memBuffer(rotbonds_unit_vectors_buf),
        RuntimeMemoryBuffer(genotype_span),
        memBuffer(output_buf)
    );

    result.dump(std::cerr);
#endif

    // Reorder result
    std::vector<InterE_AtomInput> output_buf_struct(myligand->num_of_atoms);
    static_assert(sizeof(output_buf_struct[0]) == 5 * sizeof(double));
    std::memcpy(output_buf_struct.data(), output_buf.data(), output_buf_struct.size() * sizeof(InterE_AtomInput));

    return output_buf_struct;
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

    const size_t numBytes = graphResult.size() * sizeof(graphResult[0]);
    if (std::memcmp(graphResult.data(), &myligand->atom_idxyzq[0], numBytes)) {
        std::cerr << "ChangeConform mismatch\n";

        /*for (size_t i = 0; i < graphResult.size(); ++i) {
            const auto& a = graphResult[i];
            const auto& b = myligand->atom_idxyzq[i];

            if (std::memcmp(&a, &b, sizeof(a))) {
                std::cerr << " Atom " << i << " mismatch: "
                          << a.m_atom_idxyzq[0] << "," << a.m_atom_idxyzq[1] << "," << a.m_atom_idxyzq[2] << "," << a.m_atom_idxyzq[3] << "," << a.m_atom_idxyzq[4]
                          << " vs "
                          << b[0] << "," << b[1] << "," << b[2] << "," << b[3] << "," << b[4]
                          << "\n";
            }
        }*/
    }

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
