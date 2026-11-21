#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "stl.hpp"

namespace sr {


struct EdgeTable;   // [S1'] edge_table.hpp (forward decl - header stays light)

// S2 orientMesh: face-winding orientation repair + quality sentinel (dual role).
// Data layout: Mesh.verts flat 3V (double), Mesh.triangles flat 3N (uint32_t) - see stl.hpp.

struct OrientStats {
    long long flipped = 0;      // faces flipped (BFS consistency + closed-CC volume fix)
    long long cc = 0;           // face-adjacency connected components (via manifold edges)
    long long conflicted = 0;   // manifold edges still same-direction after orient (Moebius-like)
    long long nm = 0;           // non-manifold edges (>= 3 incident faces)
    long long open = 0;         // boundary edges (1 incident face)
    long long degen = 0;        // degenerate faces (repeated index / zero area; counted, not removed)
    long long dup = 0;          // duplicate face PAIRS (same vertex set on a manifold edge; counted)
    long long inward = 0;       // closed CCs with negative signed volume (fixed when direct=true)
    long long glued = 0;        // vertices shared across CCs with opposing normals
};

// Repair in place (swaps tri[3t+1]/tri[3t+2]). direct=true also flips closed CCs
// with negative signed volume so the mesh is outward-oriented. Prints one
// "[stats] orient ..." line to stderr (chain ledger token).
OrientStats orientMesh(Mesh& mesh, bool direct = true);

// Per-face orientation against a reference-normals sidecar (written by
// hybrid_cli: "NRM1\n" + uint32 count + count*(x,y,z,nx,ny,nz) float32).
// For each face whose 3 vertices all have reference entries: flip the face if
// its winding-derived normal disagrees with the mean reference normal. This
// bypasses winding topology entirely (fixes odd-cycle regions the BFS cannot)
// and reaches ground-truth accuracy wherever the reference does. Returns the
// number of faces flipped; -1 if the file cannot be loaded.
long long orientByReference(Mesh& mesh, const std::string& nrmPath,
                             EdgeTable* et = nullptr);   // [S1'] shared table (nullptr = legacy local build)

// Embedded variant: uses Mesh::vnormals (PLY-carried, index-aligned) as the
// reference - no coordinate matching, no borrowing. Preferred when present.
long long orientByReferenceEmbedded(Mesh& mesh, EdgeTable* et = nullptr);   // [S1']

// Read-only verification on a copy (no flips): a healthy mesh must give
// flipped==0 && conflicted==0 && inward==0 (idempotency assertion).
OrientStats orientVerify(const Mesh& mesh);

} // namespace sr
