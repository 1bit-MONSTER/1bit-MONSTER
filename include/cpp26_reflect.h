// cpp26_reflect.h — P2996 (C++26 reflection) helpers for enum↔name tables.
//
// Issue #1956: adopt the toolchain-shipped C++26 subset incrementally,
// starting with the BackendType ↔ backend_name() duplication class. This
// header lets an enum's display-name table live right next to the enum
// (single source of truth), while any g++-16 -std=c++26 -freflection TU that
// includes the table gets a *compile-time proof* that the table covers every
// enumerator. Adding an enumerator without a table row is now a compile error
// instead of a silent "none"/"?"/fallback string.
//
// Compiler surface:
//   - g++-16+ with -std=c++26 -freflection  → RCPP26_HAS_REFLECTION 1, checks
//     active (requires libstdc++ 16 for <meta>, guarded below).
//   - everything else (clang/amdclang, libstdc++ 15, g++ without
//     -freflection) → RCPP26_HAS_REFLECTION 0; macros expand to nothing, no
//     new includes, no runtime cost, behavior identical to before.
//
// Usage, next to the enum it describes:
//
//   inline constexpr rcpp26::EnumName<BackendType> kBackendTypeNames[] = {
//       {BackendType::NONE, "none"},
//       {BackendType::HIP_GPU, "HIP GPU (ROCm)"},
//       ...
//   };
//   inline const char* backend_name(BackendType t) {
//       if (const char* n = rcpp26::lookup_enum_name(kBackendTypeNames, t))
//           return n;
//       return "none";
//   }
//   RCPP26_REQUIRE_ENUM_TABLE(rcpp26_verify_backend_type_names, BackendType,
//       kBackendTypeNames, "BackendType: new enumerator missing a row in "
//       "kBackendTypeNames (include/common.h)");
//
#pragma once

#include <cstddef>
#include <utility>

namespace rcpp26 {

// Table row type. Every enum→name table in the tree is an array of these so
// the reflection checks below stay generic.
template <typename E>
struct EnumName {
    E        type;
    const char* name;
};

// O(rows) lookup — tables are tiny (≤ 18 rows) and the *_name() helpers are
// not hot; this replaces hand-maintained switch statements with a table that
// reflection can inspect.
template <typename E, std::size_t N>
constexpr const char* lookup_enum_name(const EnumName<E> (&table)[N], E v) {
    for (std::size_t i = 0; i < N; ++i)
        if (table[i].type == v) return table[i].name;
    return nullptr;
}

}  // namespace rcpp26

#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202506L && \
    __has_include(<meta>)
#include <meta>
#define RCPP26_HAS_REFLECTION 1

namespace rcpp26_detail {

template <typename E, std::size_t N>
consteval bool has_name(const rcpp26::EnumName<E> (&table)[N], E v) {
    for (std::size_t i = 0; i < N; ++i)
        if (table[i].type == v) return true;
    return false;
}

}  // namespace rcpp26_detail

// Expands to a consteval verifier + static_assert: every enumerator of E must
// have a row in TABLE. MSG is the literal shown on failure — name the enum and
// the table so the next editor knows what to update. Compile-time only.
#define RCPP26_REQUIRE_ENUM_TABLE(FN, E, TABLE, MSG)                              \
    namespace rcpp26_detail {                                                     \
        consteval bool FN() {                                                     \
            constexpr static auto kEnums_ =                                       \
                std::define_static_array(std::meta::enumerators_of(^^E));         \
            template for (constexpr auto e : kEnums_) {                           \
                constexpr E v = [:e:];                                            \
                static_assert(rcpp26_detail::has_name(TABLE, v), MSG);            \
            }                                                                     \
            return true;                                                          \
        }                                                                         \
    }                                                                             \
    static_assert(rcpp26_detail::FN(), MSG);

#else  // no P2996-capable compiler in this TU
#define RCPP26_HAS_REFLECTION 0
#define RCPP26_REQUIRE_ENUM_TABLE(FN, E, TABLE, MSG) /* check inactive */ ;
#endif
