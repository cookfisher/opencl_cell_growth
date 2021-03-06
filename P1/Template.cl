/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void Add(read_only image2d_t imageA, read_only image2d_t imageB, write_only image2d_t imageC)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;
    uint B = read_imageui(imageB, sampler, (int2)(x, y)).x;

    write_imageui(imageC, (int2)(x, y), A + B);
}

__kernel void moveA(read_only image2d_t imageA, write_only image2d_t imageB, int ix, int iy)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;

    write_imageui(imageB, (int2)(x, y), A + ix);
}

__kernel void moveB(read_only image2d_t imageA, write_only image2d_t imageC, int ix, int iy)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;

    write_imageui(imageC, (int2)(x, y), A + iy);
}

__kernel void moveC(read_only image2d_t imageA, write_only image2d_t imageD, int ix, int iy)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;

    write_imageui(imageD, (int2)(x, y), iy - A);
}

__kernel void moveD(read_only image2d_t imageA, write_only image2d_t imageE, int ix, int iy)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint A = read_imageui(imageA, sampler, (int2)(x, y)).x;

    write_imageui(imageE, (int2)(x, y), ix - A);
}

__kernel void move256(__global int* A, __global int* C, __global int* D, __global int* E, __global int* F, int ix, int iy) {
    int i = get_global_id(0);
    C[i] = A[i] + ix;
    D[i] = A[i] + iy;
    E[i] = ix - A[i];
    F[i] = iy - A[i];
}