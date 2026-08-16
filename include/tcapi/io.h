// tcapi/io.h
// Mirrors tcapi_numpy/io.py.
// Sec. C2c — tensor persistence to/from a named storage string.
#pragma once

#include "itensor/all.h"
#include <string>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"

namespace tcapi {

// --- load ---------------------------------------------------------------
// Sec. C2c — read a tensor back from the storage identified by strg.
template<typename TenT, typename Storage>
tent_t<TenT> load(const context_handle_t<TenT>& ctx, const Storage& strg)
{
    detail::ensure_active<TenT>(ctx);
    tent_t<TenT> a;
    itensor::readFromFile(std::string(strg), a);
    return a;
}

// --- save ---------------------------------------------------------------
// Sec. C2c — persist tensor a to the storage identified by strg.
template<typename TenT, typename Storage>
void save(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a, const Storage& strg)
{
    detail::ensure_active<TenT>(ctx);
    itensor::writeToFile(std::string(strg), a);
}

} // namespace tcapi
