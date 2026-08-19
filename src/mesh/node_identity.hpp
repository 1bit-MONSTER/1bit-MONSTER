// node_identity.hpp — persistent node identity + capability card.
#pragma once

#include <string>

#include "mesh/mesh.hpp"

namespace mesh {

// Generate a version-4 UUID (from /dev/urandom or std::random_device).
std::string generate_uuid();

// Hostname of this machine (fallback "1bit-node").
std::string default_hostname();

// Default directory for the persistent node identity
// ($XDG_CACHE_HOME/1bit-mesh, else $HOME/.cache/1bit-mesh, else /tmp/1bit-mesh).
std::string default_state_dir();

// Load the persisted identity from <state_dir>/node.json or create + persist
// a fresh one. Never throws; falls back to a fresh in-memory identity when
// the state dir is not writable (still fully functional for the session).
NodeIdentity load_or_create_identity(const MeshConfig& cfg);

// Build api_base ("http://host:port/v1") for an identity.
std::string make_api_base(const std::string& host, uint16_t port);

// Best-effort detection of this machine's primary LAN IP (UDP connect trick,
// no packets sent). Falls back to "127.0.0.1".
std::string detect_local_ip();

} // namespace mesh
