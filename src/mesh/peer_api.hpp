// peer_api.hpp — /v1/mesh/* HTTP handlers (registered on an httplib server).
//
//   GET  /v1/mesh/me        — this node's identity card
//   GET  /v1/mesh/peers     — live peer registry
//   POST /v1/mesh/handshake — capability exchange ("hook up"); body: {node: {...}}
//   POST /v1/mesh/ask       — deliver a question;      body: {from, ask_id, type, question, node}
//   POST /v1/mesh/answer    — reply to an ask;         body: {ask_id, from, answer, accept}
//   GET  /v1/mesh/asks      — inbound ask log
//
// These are the wire surface other 1bit-MONSTER installs (and the DSH
// harness plugin, integrations/dsh) talk to. Schema: docs/mesh-protocol.md.
#pragma once

#include <httplib.h>

#include "mesh/mesh.hpp"
#include "mesh/peer_discovery.hpp"

namespace mesh {

class MeshAgent;  // fwd

// Register all /v1/mesh/* handlers on svr. Ownership of disc/agent stays
// with the caller (they must outlive the server).
void register_mesh_handlers(httplib::Server& svr, PeerDiscovery& disc,
                            MeshAgent* agent);

} // namespace mesh
