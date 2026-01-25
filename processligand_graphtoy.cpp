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

// Acceptable mismatch between original algorithm and HW implementation
static constexpr double s_intraE_tolerance_rel = 1e-10;
static constexpr double s_interE_tolerance_rel = 1e-10;
static constexpr double s_changeConform_tolerance_rel = 1e-10;

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
    KernelReadPort<double> atom_idxyzq_in,
    KernelWritePort<IntraE_AtomPair> atom_data_out
) {
    double atom_idxyzq[g_maxNumAtoms][5];
#pragma HLS bind_storage variable=atom_idxyzq type=ram_2p impl=bram

    uint32_t num_atoms = co_await num_atoms_in.get();
    num_atoms = std::min(num_atoms, uint32_t(g_maxNumAtoms));

    // Read the entire atom_idxyzq stream into local memory (cache)
    for (uint32_t idx_atom = 0; idx_atom < num_atoms; ++idx_atom) {
        for (size_t i = 0; i < 5; ++i) {
            atom_idxyzq[idx_atom][i] = co_await atom_idxyzq_in.get();
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

/**
 * Reads atom_idxyzq from memory sequentially and streams it out as doubles.
 * Output order: atom0[0..4], atom1[0..4], ...
 */
COMPUTE_KERNEL(hls, kernel_StreamAtomIdxyzq,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelMemoryPort<const double> atom_idxyzq_buf,
    KernelWritePort<double> atom_idxyzq_out
) {
    uint32_t num_atoms = co_await num_atoms_in.get();

    num_atoms = std::min(num_atoms, uint32_t(g_maxNumAtoms));

    for (uint32_t idx_atom = 0; idx_atom < num_atoms; ++idx_atom) {
#pragma HLS unroll off
        for (uint32_t i = 0; i < 5; ++i) {
#pragma HLS unroll off
            co_await atom_idxyzq_out.put(atom_idxyzq_buf[idx_atom * 5 + i]);
        }
    }
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

static constexpr void build_intraE_core(
    IoConnector<uint32_t> num_atoms_in,
    IoConnector<const char> intraE_contributors_buf,
    IoConnector<double> atom_idxyzq_stream,
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

    kernel_IntraE_GenAtomPairIndices(num_atoms_in, intraE_contributors_buf, atom_pairs);
    kernel_IntraE_FetchAtomData(num_atoms_in, atom_pairs, atom_idxyzq_stream, atom_data_fetched);
    kernel_IntraE_SetDistanceID_CheckHBond(atom_data_fetched, atom_data_distance_checked, dcutoff_in, is_hbond_lut_buf);
    kernel_IntraE_Volume_Solpar(atom_data_distance_checked, atom_data_with_volume, qasp_in, volume_buf, solpar_buf);
    kernel_IntraE_FetchVWpars(atom_data_with_volume, atom_data_with_vw_fetched, vwpars_a_buf, vwpars_b_buf, vwpars_c_buf, vwpars_d_buf);
    kernel_IntraE_ScaleVWparsWithDistance(atom_data_with_vw_fetched, atom_data_with_vw_scaled);
    kernel_IntraE_Compute_VW_EL_Desolv(atom_data_with_vw_scaled, scaled_AD4_coeff_elec_in, AD4_coeff_desolv_in, out_buf);
}

COMPUTE_KERNEL(hls, kernel_ScaleLigandAtomIdxyzq,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelReadPort<double, s_iop_rtp> scale_factor_in,
    KernelReadPort<double> atom_idxyzq_in,
    KernelWritePort<double> atom_idxyzq_out
) {
    const uint32_t num_atoms = co_await num_atoms_in.get();
    const double scale_factor = co_await scale_factor_in.get();

    for (uint32_t atom_idx = 0; atom_idx < num_atoms; ++atom_idx) {
#pragma HLS unroll off
        for (uint32_t i = 0; i < 5; ++i) {
#pragma HLS unroll off
            double val = co_await atom_idxyzq_in.get();
            if (i >= 1 && i < 4) {
                val *= scale_factor;
            }
            co_await atom_idxyzq_out.put(val);
        }
    }
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
    return calc_intraE_original(myligand, dcutoff, ignore_desolv, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp, debug);
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
    KernelReadPort<double> atom_idxyzq_in,
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
            inputData[j] = co_await atom_idxyzq_in.get();
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

/**
 * Reads atom_idxyzq from memory sequentially and streams it out as doubles.
 * Output order: atom0[0..4], atom1[0..4], ...
 */

static constexpr uint32_t s_terminate_dram_reader_sentinel = UINT32_MAX;

static constexpr uint32_t s_interE_coords_per_cube = 8;
static constexpr uint32_t s_interE_grids_per_atom = 3;
static constexpr uint32_t s_interE_addresses_per_atom = s_interE_coords_per_cube * s_interE_grids_per_atom;

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

        std::array<int, s_interE_coords_per_cube> coordOffsets = {
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_low, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_low, data.m_x_high),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_high, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_low, data.m_y_high, data.m_x_high),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_low, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_low, data.m_x_high),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_high, data.m_x_low),
            interE_gridCoordsToArrayOffset(size_xyz, data.m_z_high, data.m_y_high, data.m_x_high)
        };

        std::array<int, s_interE_grids_per_atom> gridNumbers = {
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

COMPUTE_KERNEL_TEMPLATE(hls, kernel_InterE_ReadGrid,
    KernelReadPort<uint32_t> address_in,
    KernelMemoryPort<const T> dram_buf,
    KernelWritePort<T> data_out
) {
    // Slightly weird pipelining setup:
    // Read a whole atom's worth of grid data into a buffer with
    // pipelined reads, then send it to the downstream kernel all
    // at once.
    //
    // This avoids a possible deadlock with full pipelining:
    // - This kernel issues a whole bunch of reads to DRAM
    // - But it can't receive the read responses until there's space in data_out
    // - This fills the memory subsystem's queues, causing it to stall
    // - The downstream kernels can't proceed because they have to access DRAM too
    // - So they'll never read from data_out
    // - Deadlock.
    //
    // Buffering the read data ensures that this kernel can always
    // accept read responses from DRAM, even if the downstream kernel
    // is stalled.

    T intermediate_buffer[s_interE_addresses_per_atom];

    while (true) {
#pragma HLS pipeline off

        for (uint32_t i = 0; i < s_interE_addresses_per_atom; ++i) {
#pragma HLS pipeline style=frp
            const uint32_t addr = co_await address_in.get();

            if ((i == 0) && (addr == s_terminate_dram_reader_sentinel)) {
                goto out;
            }

            intermediate_buffer[i] = dram_buf[addr];
        }

        for (uint32_t i = 0; i < s_interE_addresses_per_atom; ++i) {
#pragma HLS pipeline style=stp
            co_await data_out.put(intermediate_buffer[i]);
        }
    }

    out:;
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

static constexpr void build_interE_core(
    IoConnector<uint32_t> num_atoms_in,
    IoConnector<double> outofgrid_tolerance_in,
    IoConnector<int> grid_size_x_in,
    IoConnector<int> grid_size_y_in,
    IoConnector<int> grid_size_z_in,
    IoConnector<int> num_of_atypes_in,
    IoConnector<bool> enable_peratom_outputs_in,
    IoConnector<double> atom_idxyzq_stream,
    IoConnector<const double> grid_buf,
    IoConnector<double> peratom_vdw_buf,
    IoConnector<double> peratom_elec_buf,
    IoConnector<double> scalar_out_buf
) {
    IoConnector<InterE_AtomData> atom_data_stream;
    IoConnector<uint32_t> dram_address_stream;
    IoConnector<double> dram_data_stream;
    IoConnector<InterE_AtomEnergy> atom_energy_stream;

    kernel_interE_BuildAtomData(
        num_atoms_in,
        outofgrid_tolerance_in,
        grid_size_x_in,
        grid_size_y_in,
        grid_size_z_in,
        atom_idxyzq_stream,
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

    kernel_InterE_ReadGrid<double>(
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
}

static auto getNumGridElems(const Gridinfo *mygrid) {
    const auto& gridsize = mygrid->size_xyz;
    return (mygrid->num_of_atypes + 2) * gridsize[0] * gridsize[1] * gridsize[2];
}

double calc_interE(const Gridinfo* mygrid, const Liganddata* myligand, const double* fgrids, double outofgrid_tolerance, int debug) {
    return calc_interE_original(mygrid, myligand, fgrids, outofgrid_tolerance, debug);
}

void calc_interE_peratom(const Gridinfo* mygrid, const Liganddata* myligand, const double* fgrids, double outofgrid_tolerance,
                         double* elecE, double peratom_vdw [256], double peratom_elec [256], int debug)
{
    calc_interE_peratom_original(mygrid, myligand, fgrids, outofgrid_tolerance, elecE, peratom_vdw, peratom_elec, debug);
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
    KernelWritePort<double> atom_idxyzq_out
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

    while (true) {
        auto data = co_await atom_data_in.get();

        if (data.m_terminate_processing) {
            break;
        }

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

        for (uint32_t i = 0; i < 5; ++i) {
#pragma HLS unroll off
            co_await atom_idxyzq_out.put(data.m_atomdata.m_atom_idxyzq[i]);
        }
    }
}

COMPUTE_KERNEL(hls, kernel_ChangeConform_WriteAtomsToMemory,
    KernelReadPort<uint32_t, s_iop_rtp> num_atoms_in,
    KernelReadPort<double> atom_idxyzq_in,
    KernelMemoryPort<double> output_buf
) {
    const uint32_t num_atoms = co_await num_atoms_in.get();

    for (uint32_t atom_idx = 0; atom_idx < num_atoms; ++atom_idx) {
#pragma HLS unroll off
        for (uint32_t i = 0; i < 5; ++i) {
#pragma HLS unroll off
            const double val = co_await atom_idxyzq_in.get();
            if (atom_idx < g_maxNumAtoms) {
                output_buf[atom_idx * 5 + i] = val;
            }
        }
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

static constexpr IoConnector<double> build_changeConform_core(
    IoConnector<uint32_t> num_atoms_in,
    IoConnector<uint32_t> num_rotbonds_in,
    IoConnector<double> initial_move_x_in,
    IoConnector<double> initial_move_y_in,
    IoConnector<double> initial_move_z_in,
    IoConnector<const double> atom_data_buf,
    IoConnector<const char> rotbonds_buf,
    IoConnector<const double> rotbonds_moving_vectors_buf,
    IoConnector<const double> rotbonds_unit_vectors_buf,
    IoConnector<const double> genotype_buf
) {
    IoConnector<ChangeConform_AtomData> atoms_read;
    IoConnector<double> genotype_sincos_precomputed;
    IoConnector<double> atom_idxyzq_stream;

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
        atom_idxyzq_stream
    );

    return atom_idxyzq_stream;
}

COMPUTE_GRAPH constexpr auto intra_interE_uber_graph = make_compute_graph_v<[] (
    // ChangeConform inputs
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

    // InterE inputs
    IoConnector<double> outofgrid_tolerance_in,
    IoConnector<int> grid_size_x_in,
    IoConnector<int> grid_size_y_in,
    IoConnector<int> grid_size_z_in,
    IoConnector<int> num_of_atypes_in,
    IoConnector<bool> enable_peratom_outputs_in,
    IoConnector<const double> grid_buf,
    IoConnector<double> peratom_vdw_buf,
    IoConnector<double> peratom_elec_buf,
    IoConnector<double> interE_scalar_out_buf,

    // IntraE inputs
    IoConnector<double> scale_factor_in,
    IoConnector<const char> intraE_contributors_buf,
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
    IoConnector<double> intraE_out_buf
) {
    IoConnector<double> atom_idxyzq_stream_scaled;

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

    CGSIM_AUTO_NAME(outofgrid_tolerance_in);
    CGSIM_AUTO_NAME(grid_size_x_in);
    CGSIM_AUTO_NAME(grid_size_y_in);
    CGSIM_AUTO_NAME(grid_size_z_in);
    CGSIM_AUTO_NAME(num_of_atypes_in);
    CGSIM_AUTO_NAME(enable_peratom_outputs_in);
    CGSIM_AUTO_NAME(grid_buf);
    CGSIM_AUTO_NAME(peratom_vdw_buf);
    CGSIM_AUTO_NAME(peratom_elec_buf);
    CGSIM_AUTO_NAME(interE_scalar_out_buf);

    CGSIM_AUTO_NAME(scale_factor_in);
    CGSIM_AUTO_NAME(intraE_contributors_buf);
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
    CGSIM_AUTO_NAME(intraE_out_buf);

    const auto atom_idxyzq_stream = build_changeConform_core(
        num_atoms_in,
        num_rotbonds_in,
        initial_move_x_in,
        initial_move_y_in,
        initial_move_z_in,
        atom_data_buf,
        rotbonds_buf,
        rotbonds_moving_vectors_buf,
        rotbonds_unit_vectors_buf,
        genotype_buf
    );

    // Only IntraE uses scaled coordinates.
    kernel_ScaleLigandAtomIdxyzq(
        num_atoms_in,
        scale_factor_in,
        atom_idxyzq_stream,
        atom_idxyzq_stream_scaled
    );

    build_interE_core(
        num_atoms_in,
        outofgrid_tolerance_in,
        grid_size_x_in,
        grid_size_y_in,
        grid_size_z_in,
        num_of_atypes_in,
        enable_peratom_outputs_in,
        atom_idxyzq_stream,
        grid_buf,
        peratom_vdw_buf,
        peratom_elec_buf,
        interE_scalar_out_buf
    );

    build_intraE_core(
        num_atoms_in,
        intraE_contributors_buf,
        atom_idxyzq_stream_scaled,
        is_hbond_lut_buf,
        volume_buf,
        solpar_buf,
        vwpars_a_buf,
        vwpars_b_buf,
        vwpars_c_buf,
        vwpars_d_buf,
        dcutoff_in,
        qasp_in,
        scaled_AD4_coeff_elec_in,
        AD4_coeff_desolv_in,
        intraE_out_buf
    );

    return std::tuple();
}>;

extern "C" void eval_intra_interE_for_genotype_graphtoy(
    const Liganddata* myligand_ref_ori,
    const double* genotype,
    const Gridinfo* myginfo,
    const double* grids,
    double interE_smooth,
    int intraE_num_of_evals,
    int ignore_desolv,
    double scaled_AD4_coeff_elec,
    double AD4_coeff_desolv,
    double qasp,
    int debug,
    double* out_intra_inter /* out[0]=intraE, out[1]=interE */
) {
    (void)debug;

    // Build required derived buffers
    auto is_hbond_lut_buf = intraE_build_hbond_lut(myligand_ref_ori);

    // InterE outputs (per-atom disabled here)
    std::vector<double> peratom_vdw_buf{1, 0.0};
    std::vector<double> peratom_elec_buf{1, 0.0};
    std::vector<double> interE_out_buf(2, 0.0); // interE, elecE

    // IntraE outputs
    std::vector<double> intraE_out_buf(3, 0.0); // vW, el, desolv

    // Initial move-to-origin for ChangeConform
    double initial_move_xyz[3];
    get_movvec_to_origo(myligand_ref_ori, initial_move_xyz);

    const auto numGridMemElems = getNumGridElems(myginfo);
    std::span<const double> gridMemoryRegion{grids, size_t(numGridMemElems)};

    const auto genotype_span = std::span<const double>(genotype, size_t(6 + myligand_ref_ori->num_of_rotbonds));

    const auto atom_idxyzq_span = std::span<const double>(&myligand_ref_ori->atom_idxyzq[0][0], size_t(g_maxNumAtoms * 5));
    const auto atom_rotbonds_span = std::span<const char>(&myligand_ref_ori->atom_rotbonds[0][0], size_t(g_maxNumAtoms * g_maxNumRotbonds));
    const auto intraE_contributors_span = std::span<const char>(&myligand_ref_ori->intraE_contributors[0][0], size_t(g_maxNumAtoms * g_maxNumAtoms));
    const auto vwpars_a_span = std::span<const double>(&myligand_ref_ori->VWpars_A[0][0], size_t(g_maxNumAtomTypes * g_maxNumAtomTypes));
    const auto vwpars_b_span = std::span<const double>(&myligand_ref_ori->VWpars_B[0][0], size_t(g_maxNumAtomTypes * g_maxNumAtomTypes));
    const auto vwpars_c_span = std::span<const double>(&myligand_ref_ori->VWpars_C[0][0], size_t(g_maxNumAtomTypes * g_maxNumAtomTypes));
    const auto vwpars_d_span = std::span<const double>(&myligand_ref_ori->VWpars_D[0][0], size_t(g_maxNumAtomTypes * g_maxNumAtomTypes));

    const auto rotbonds_moving_vectors_span = std::span<const double>(&myligand_ref_ori->rotbonds_moving_vectors[0][0], size_t(g_maxNumRotbonds * 3));
    const auto rotbonds_unit_vectors_span = std::span<const double>(&myligand_ref_ori->rotbonds_unit_vectors[0][0], size_t(g_maxNumRotbonds * 3));

#ifdef HAVE_TAPASCO
    // TaPaSCo backend for the uber-graph isn't wired up yet.
    // Fall back to the existing separate kernels/graphs.
    Liganddata myligand_temp = *myligand_ref_ori;
    change_conform(&myligand_temp, genotype, debug);
    out_intra_inter[1] = calc_interE(myginfo, &myligand_temp, grids, interE_smooth, debug);
    scale_ligand(&myligand_temp, myginfo->spacing);
    out_intra_inter[0] = calc_intraE(&myligand_temp, intraE_num_of_evals, ignore_desolv, scaled_AD4_coeff_elec, AD4_coeff_desolv, qasp, debug);
    return;
#else
    const auto result = intra_interE_uber_graph(
        ScalarDataSource<uint32_t>(myligand_ref_ori->num_of_atoms),
        ScalarDataSource<uint32_t>(myligand_ref_ori->num_of_rotbonds),
        ScalarDataSource<double>(initial_move_xyz[0]),
        ScalarDataSource<double>(initial_move_xyz[1]),
        ScalarDataSource<double>(initial_move_xyz[2]),
        RuntimeMemoryBuffer(atom_idxyzq_span),
        RuntimeMemoryBuffer(atom_rotbonds_span),
        RuntimeMemoryBuffer(rotbonds_moving_vectors_span),
        RuntimeMemoryBuffer(rotbonds_unit_vectors_span),
        RuntimeMemoryBuffer(genotype_span),

        ScalarDataSource<double>(interE_smooth),
        ScalarDataSource<int>(myginfo->size_xyz[0]),
        ScalarDataSource<int>(myginfo->size_xyz[1]),
        ScalarDataSource<int>(myginfo->size_xyz[2]),
        ScalarDataSource<int>(myligand_ref_ori->num_of_atypes),
        ScalarDataSource<bool>(false),
        RuntimeMemoryBuffer(gridMemoryRegion),
        memBuffer(peratom_vdw_buf),
        memBuffer(peratom_elec_buf),
        memBuffer(interE_out_buf),

        ScalarDataSource<double>(myginfo->spacing),
        RuntimeMemoryBuffer(intraE_contributors_span),
        memBuffer(is_hbond_lut_buf),
        RuntimeMemoryBuffer(std::span<const double>(myligand_ref_ori->volume, g_maxNumAtomTypes)),
        RuntimeMemoryBuffer(std::span<const double>(myligand_ref_ori->solpar, g_maxNumAtomTypes)),
        RuntimeMemoryBuffer(vwpars_a_span),
        RuntimeMemoryBuffer(vwpars_b_span),
        RuntimeMemoryBuffer(vwpars_c_span),
        RuntimeMemoryBuffer(vwpars_d_span),
        ScalarDataSource<double>(double(intraE_num_of_evals)),
        ScalarDataSource<double>(qasp),
        ScalarDataSource<double>(scaled_AD4_coeff_elec),
        ScalarDataSource<double>(AD4_coeff_desolv),
        memBuffer(intraE_out_buf)
    );

    // Warn about deadlocks
    result.dump(std::cerr);

    const double vW = intraE_out_buf[0];
    const double el = intraE_out_buf[1];
    const double desolv = intraE_out_buf[2];

    out_intra_inter[1] = interE_out_buf[0];
    out_intra_inter[0] = vW + el + (ignore_desolv ? 0.0 : desolv);
#endif
}

void change_conform(Liganddata* myligand, const double genotype [], int debug) {
    change_conform_original(myligand, genotype, debug);
}

extern "C" void enable_graphdumps() {
    g_graphdumpsEnabled = true;
}
