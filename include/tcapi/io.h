#pragma once

#include "itensor/all.h"
#include <string>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"

namespace tcapi {

template<typename TenT, typename Storage>
ten_t<TenT> load(context_handle_t<TenT>& ctx, const Storage& strg)
{
    detail::ensure_active<TenT>(ctx);
    ten_t<TenT> a;
    itensor::readFromFile(std::string(strg), a);
    return a;
}

template<typename TenT, typename Storage>
void save(context_handle_t<TenT>& ctx, const ten_t<TenT>& a, const Storage& strg)
{
    detail::ensure_active<TenT>(ctx);
    itensor::writeToFile(std::string(strg), a);
}

} // namespace tcapi