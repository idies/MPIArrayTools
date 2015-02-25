#include <mpi.h>
#include <stdarg.h>
#include <iostream>

#ifndef BASE

#define BASE

static const int message_buffer_length = 1024;
static char debug_message_buffer[message_buffer_length];
extern int myrank, nprocs;

#ifndef NDEBUG

inline void DEBUG_MSG(const char * format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    sprintf(
            debug_message_buffer,
            "cpu%.4d ",
            myrank);
    vsnprintf(
            debug_message_buffer + 8,
            message_buffer_length - 8,
            format,
            argptr);
    va_end(argptr);
    std::cerr << debug_message_buffer;
}

inline void DEBUG_MSG_WAIT(MPI_Comm communicator, const char * format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    sprintf(
            debug_message_buffer,
            "cpu%.4d ",
            myrank);
    vsnprintf(
            debug_message_buffer + 8,
            message_buffer_length - 8,
            format,
            argptr);
    va_end(argptr);
    std::cerr << debug_message_buffer;
    MPI_Barrier(communicator);
}

#else

#define DEBUG_MSG(...)
#define DEBUG_MSG_WAIT(...)

#endif//NDEBUG

#endif//BASE

