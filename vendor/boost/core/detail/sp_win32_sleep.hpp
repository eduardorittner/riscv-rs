#ifndef BOOST_CORE_DETAIL_SP_WIN32_SLEEP_HPP_INCLUDED
#define BOOST_CORE_DETAIL_SP_WIN32_SLEEP_HPP_INCLUDED

// MS compatible compilers support #pragma once

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

// boost/core/detail/sp_win32_sleep.hpp
//
// Provides the Win32 declarations for Sleep() and SwitchToThread()
// used by sp_thread_sleep.hpp and sp_thread_yield.hpp when building
// on Windows (including the MinGW-w64 cross-compiler).
//
// Copyright 2008, 2020 Peter Dimov
// Distributed under the Boost Software License, Version 1.0
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/config.hpp>

#if defined( BOOST_USE_WINDOWS_H )
# include <windows.h>
#endif

namespace boost
{
namespace detail
{

#if !defined( BOOST_USE_WINDOWS_H )

extern "C" __declspec(dllimport) void __stdcall Sleep( unsigned long ms );

extern "C" __declspec(dllimport) int __stdcall SwitchToThread( void );

#endif

} // namespace detail
} // namespace boost

#endif // #ifndef BOOST_CORE_DETAIL_SP_WIN32_SLEEP_HPP_INCLUDED
