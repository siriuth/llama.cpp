#include "binbcast.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "ggml.h"
#include "siriuth.hpp"

//#define SYCL_BCAST_WORK_GROUP_NUM 1
//#define SYCL_BCAST_WORK_GROUP_NUM 64
//#define SYCL_BCAST_WORK_GROUP_NUM 128
#define SYCL_BCAST_WORK_GROUP_NUM 999999
//#define SYCL_BCAST_WORK_GROUP_NUM 64
//#define SYCL_BCAST_WORK_GROUP_NUM 16
//#define SYCL_BCAST_WORK_GROUP_SIZE 16
#define SYCL_BCAST_WORK_GROUP_SIZE 32
//#define SYCL_BCAST_WORK_GROUP_SIZE 64
//#define SYCL_BCAST_WORK_GROUP_SIZE 128
//#define SYCL_BCAST_WORK_GROUP_SIZE 256
//#define SYCL_BCAST_WORK_GROUP_SIZE 64
//#define SYCL_BCAST_SUB_GROUP_SIZE 8
#define SYCL_BCAST_SUB_GROUP_SIZE 16
//#define SYCL_BCAST_SUB_GROUP_SIZE 32
//#define SYCL_BCAST_SUB_GROUP_SIZE 64 // Sub-group size 64 is not supported on the deviceException

#define SYCL_OTHER_WORK_GROUP_NUM 8

/*
template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        //int s0,
        int s1,  int s2,  int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0*s00] : 0.0f, (float)src1_row[i10*s10]);
    }
}
*/


// ne2とne3をまとめた形で1個実行する。
template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_one(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        int s0,  int s1,  int s2,  int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        const sycl::range<3> offset,
        const sycl::nd_item<3> &item_ct1) {
    const int i0 = item_ct1.get_global_id(2) + offset[2];
    const int i1 = item_ct1.get_global_id(1) + offset[1];
    const int i2 = (item_ct1.get_global_id(0) + offset[0]) / ne3;
    const int i3 = (item_ct1.get_global_id(0) + offset[0]) % ne3;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2) {
        return;
    }
    /*
    const int i10 = i0 % ne10;
    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01 +  i0*s00;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11 + i10*s10;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1  +  i0*s0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;
    */
    // 変数を排除して即値形式にしてみたが、コンパイラの最適化で速度は変わらない感じ。
    // 可読性が失われるので、元の形のソースを生かしておく。
    const src0_t * src0_row = src0 + i3*s03 +  i2*s02 +  i1*s01 +  i0*s00;
    const src1_t * src1_row = src1 + (i3 % ne13)*s13 + (i2 % ne12)*s12 + (i1 % ne11)*s11 + (i0 % ne10)*s10;
    dst_t * dst_row = dst + i3*s3  +  i2*s2  +  i1*s1  +  i0*s0;

    //dst_row[0] = (dst_t)bin_op(src0 ? (float)src0_row[0] : 0.0f, (float)src1_row[0]);
    *(dst_row) = (dst_t)bin_op(src0 ? (float)*(src0_row) : 0.0f, (float)*(src1_row));
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_one_sd(const src0_t * src0, const src1_t * src1, dst_t * dst,
        const int* ne,
        const int* ne1,
        const int* s,
        const int* s0,
        const int* s1,
        const sycl::range<3> offset,
        const sycl::nd_item<3> &item_ct1) {
    const int i0 = item_ct1.get_global_id(2) + offset[2];
    const int i1 = item_ct1.get_global_id(1) + offset[1];
    const int i2 = (item_ct1.get_global_id(0) + offset[0]) / ne[3];
    const int i3 = (item_ct1.get_global_id(0) + offset[0]) % ne[3];

    if (i0 >= ne[0] || i1 >= ne[1] || i2 >= ne[2]) {
        return;
    }
    const src0_t * src0_row = src0 + i3*s0[3] +  i2*s0[2] +  i1*s0[1] +  i0*s0[0];
    const src1_t * src1_row = src1 + (i3 % ne1[3])*s1[3] + (i2 % ne1[2])*s1[2] + (i1 % ne1[1])*s1[1] + (i0 % ne1[0])*s1[0];
    dst_t * dst_row = dst + i3*s[3]  +  i2*s[2]  +  i1*s[1]  +  i0*s[0];

    *(dst_row) = (dst_t)bin_op(src0 ? (float)*(src0_row) : 0.0f, (float)*(src1_row));
}

// 3次元まで。
template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_one_u3d(const src0_t * src0, const src1_t * src1, dst_t * dst,
        const int* ne,
        const int* ne1,
        const int* s,
        const int* s0,
        const int* s1,
        const sycl::range<3> offset,
        const sycl::nd_item<3> &item_ct1) {
    int i[3] = {
        item_ct1.get_global_id(2) + offset[2],
        item_ct1.get_global_id(1) + offset[1],
        item_ct1.get_global_id(0) + offset[0]};

    if (i[0] >= ne[0] || i[1] >= ne[1] || i[2] >= ne[2]) {
        return;
    }
    const src0_t * src0_row = src0 + i[2]*s0[2] +  i[1]*s0[1] +  i[0]*s0[0];
    const src1_t * src1_row = src1 + (i[2] % ne1[2])*s1[2] + (i[1] % ne1[1])*s1[1] + (i[0] % ne1[0])*s1[0];
    dst_t * dst_row = dst + i[2]*s[2]  +  i[1]*s[1]  +  i[0]*s[0];

    //dst_row[0] = (dst_t)bin_op(src0 ? (float)src0_row[0] : 0.0f, (float)src1_row[0]);
    *(dst_row) = (dst_t)bin_op(src0 ? (float)*(src0_row) : 0.0f, (float)*(src1_row));
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(
        const src0_t * src0,
        const src1_t * src1,
        dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0,*/ int s1,  int s2,  int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        int k,
        int offset,
        const sycl::nd_item<1> &item_ct1) {

    //const int i = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
    //              item_ct1.get_local_id(0) + offset;
    const int i = item_ct1.get_global_id(0) + offset;

    if (i >= k) {
        return;
    }

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    //if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
    if (i3 >= ne3) {
        return;
    }

    const int i10 = i0 % ne10;
    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0*s00] : 0.0f, (float)src1_row[i10*s10]);
}

/*
template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_src1_unravel(
        const src0_t * src0,
        const src1_t * src1,
        dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        //int s0,
        int s1,  int s2,  int s3,
        int s10, int s11, int s12, int s13,
        int k,
        int offset,
        const sycl::nd_item<1> &item_ct1) {
    const int i = item_ct1.get_global_id(0) + offset;

    if (i >= k) {
        return;
    }

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i3 >= ne3) {
        return;
    }

    const int i10 = i0 % ne10;
    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    dst_row[i0] = (dst_t)bin_op(0.0f, (float)src1_row[i10*s10]);

    GGML_UNUSED(src0);
}
*/

template<float (*bin_op)(const float, const float)>
struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
            const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
            const int64_t ne0,  const int64_t ne1,  const int64_t ne2,  const int64_t ne3,
            const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
            const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13,
            const size_t nb0,  const size_t nb1,  const size_t nb2,  const size_t nb3,
            const bool src0_is_contiguous, const bool src1_is_contiguous,
            const bool src0_is_permuted, const bool src1_is_permuted,
            queue_ptr stream) {

        int nr0 = ne10 / ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[]  = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[]  = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }
        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

/*
            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned int>(hne0, block_size);
            block_dims[1] = std::min<unsigned int>(
                ne1, block_size / (unsigned int)block_dims[2]);
            block_dims[0] = std::min(
            std::min<unsigned int>(ne2 * ne3, block_size / (unsigned int)block_dims[2] / (unsigned int)block_dims[1]), 64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            sycl::device dev = stream->get_device();
            const int64_t max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
            GGML_SYCL_DEBUG("[SYCL] %s max_work_group_size:%ld\n", __func__, max_work_group_size);

                    // 旧ロジック
                    stream->parallel_for(
                        sycl::nd_range<3>(block_nums * block_dims, block_dims),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast<bin_op>(
                                src0_dd, src1_dd, dst_dd,
                                ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13,
                                s1, s2, s3,
                                s00, s01, s02, s03,
                                s10, s11, s12, s13,
                                item_ct1);
                    });

*/
            int sd[4] = {0, 1, 2, 3};
            const int64_t tne[4] = {ne0, ne1, ne2, ne3};
            const size_t tnb[4] = {nb0, nb1, nb2, nb3};
            //ggml_sortDim(sd, tne);
            ggml_sortDim(sd, tne, tnb);

            GGML_SYCL_DEBUG("[SYCL] %s k_bin_bcast_one ne(%ld, %ld, %ld, %ld) sd(%d, %d, %d, %d)\n", __func__, ne0, ne1, ne2, ne3, sd[0], sd[1], sd[2], sd[3]);

            // 3次元以下 under three dim.
            const int64_t tne1[4] = {ne10, ne11, ne12, ne13};
            const size_t ts[4] = {s0, s1, s2, s3};
            const size_t ts0[4] = {s00, s01, s02, s03};
            const size_t ts1[4] = {s10, s11, s12, s13};

            int ane[4];
            int ane1[4];
            int as[4];
            int as0[4];
            int as1[4];
            for(int i=0; i<4; i++){
                ane[i]  = tne[sd[i]];
                ane1[i] = tne1[sd[i]];
                as[i]   = ts[sd[i]];
                as0[i]  = ts0[sd[i]];
                as1[i]  = ts1[sd[i]];
            }
/*
            if(((ne0==1?0:1)+(ne1==1?0:1)+(ne2==1?0:1)+(ne3==1?0:1)) <4){

            sycl::range<3> world(ane[2], ane[1], ane[0]);
            sycl::range<3> local(1, 1, SYCL_BCAST_WORK_GROUP_SIZE);
            ggml_sycl_looper(world, local, SYCL_BCAST_WORK_GROUP_NUM, stream,
                [=](sycl::range<3> global, sycl::range<3> offset){

                    auto e =
                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                        [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(SYCL_BCAST_SUB_GROUP_SIZE)]]
                        {
                            k_bin_bcast_one_u3d<bin_op>(
                                src0_dd, src1_dd, dst_dd,
                                ane,
                                ane1,
                                as,
                                as0,
                                as1,
                                offset,
                                item_ct1);
                        }
                    );
                    SyclQueueEventWatcher::getInstance().SetEvent(e);
                }
            );

            }else{
*/
            // 4次元
/*
            sycl::range<3> world(ne3*ne2, ne1, ne0);
            sycl::range<3> local(1, 1, SYCL_BCAST_WORK_GROUP_SIZE);
            ggml_sycl_looper(world, local, SYCL_BCAST_WORK_GROUP_NUM, stream,
                [=](sycl::range<3> global, sycl::range<3> offset){

                    auto e =
                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                        [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(SYCL_BCAST_SUB_GROUP_SIZE)]]
                        {
                            k_bin_bcast_one<bin_op>(
                                src0_dd, src1_dd, dst_dd,
                                ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13,
                                s0, s1, s2, s3,
                                s00, s01, s02, s03,
                                s10, s11, s12, s13,
                                offset,
                                item_ct1);
                        }
                    );
                    SyclQueueEventWatcher::getInstance().SetEvent(e);
                }
            );
*/



            sycl::range<3> world(ane[3]*ane[2], ane[1], ane[0]);
            //sycl::range<3> local(1, 1, SYCL_BCAST_WORK_GROUP_SIZE);
            sycl::range<3> local;
            adjustment_local(local, world, SYCL_BCAST_WORK_GROUP_SIZE, SYCL_BCAST_SUB_GROUP_SIZE);
            ggml_sycl_looper(world, local, SYCL_BCAST_WORK_GROUP_NUM, stream,
                [=](sycl::range<3> global, sycl::range<3> offset){

                    auto e =
                        stream->parallel_for(
                            sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(SYCL_BCAST_SUB_GROUP_SIZE)]]
                            {
                                k_bin_bcast_one_sd<bin_op>(
                                    src0_dd, src1_dd, dst_dd,
                                    ane,
                                    ane1,
                                    as,
                                    as0,
                                    as1,
                                    offset,
                                    item_ct1);
                            }
                                );
                    SyclQueueEventWatcher::getInstance().SetEvent(e);
                }
            );



            //            }

        }
    }
};

template <class op>
inline void ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_BINARY_OP_LOCALS

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((const float *) src0->data, (const float *) src1->data, (float *) dst->data, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0->data, (const sycl::half *) src1->data, (sycl::half *) dst->data, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0->data, (const float *) src1->data, (sycl::half *) dst->data, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((const int32_t *) src0->data, (const int32_t *) src1->data, (int32_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((const int16_t *) src0->data, (const int16_t *) src1->data, (int16_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((const sycl::ext::oneapi::bfloat16 *) src0->data, (const sycl::ext::oneapi::bfloat16 *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((const sycl::ext::oneapi::bfloat16 *) src0->data, (const float *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
#endif
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void k_bin_bcast3(const src0_t * src0, const src1_t * src1, const src2_t * src2, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        int ne20, int ne21, int ne22, int ne23,
        int s1, int s2, int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        int s20, int s21, int s22, int s23,
        const sycl::nd_item<3> & item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;
    const int i21 = i1 % ne21;
    const int i22 = i2 % ne22;
    const int i23 = i3 % ne23;

    const size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    const size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    const src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int   i10 = i0 % ne10;
        const int   i20 = i0 % ne20;
        const float acc = bin_op((float) src0_row[i0 * s00], (float) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (float) src2_row[i20 * s20]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void k_bin_bcast3_unravel(const src0_t * src0, const src1_t * src1, const src2_t * src2, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        int ne20, int ne21, int ne22, int ne23,
        int s1, int s2, int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        int s20, int s21, int s22, int s23,
        const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    const int i3 = i / (ne2 * ne1 * ne0);
    const int i2 = (i / (ne1 * ne0)) % ne2;
    const int i1 = (i / ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;
    const int i21 = i1 % ne21;
    const int i22 = i2 % ne22;
    const int i23 = i3 % ne23;

    const size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    const size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    const int   i10 = i0 % ne10;
    const int   i20 = i0 % ne20;
    const float acc = bin_op((float) src0[i_src0 + i0 * s00], (float) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (float) src2[i_src2 + i20 * s20]);
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void launch_bin_bcast3(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    int nr1[4] = { (int) (ne10 / ne0), (int) (ne11 / ne1), (int) (ne12 / ne2), (int) (ne13 / ne3) };
    int nr2[4] = { (int) (ne20 / ne0), (int) (ne21 / ne1), (int) (ne22 / ne2), (int) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    const bool can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    if (can_collapse) {
        for (int i = 0; i < 4; i++) {
            if (nr1[i] != 1 || nr2[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse_nb(cnb2, cne2);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
                collapse(cne2);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        int64_t ne10 = cne1[0];
        int64_t ne11 = cne1[1];
        int64_t ne12 = cne1[2];
        int64_t ne13 = cne1[3];

        int64_t ne20 = cne2[0];
        int64_t ne21 = cne2[1];
        int64_t ne22 = cne2[2];
        int64_t ne23 = cne2[3];

        size_t s1 = cnb[1] / sizeof(dst_t);
        size_t s2 = cnb[2] / sizeof(dst_t);
        size_t s3 = cnb[3] / sizeof(dst_t);

        size_t s00 = cnb0[0] / sizeof(src0_t);
        size_t s01 = cnb0[1] / sizeof(src0_t);
        size_t s02 = cnb0[2] / sizeof(src0_t);
        size_t s03 = cnb0[3] / sizeof(src0_t);

        size_t s10 = cnb1[0] / sizeof(src1_t);
        size_t s11 = cnb1[1] / sizeof(src1_t);
        size_t s12 = cnb1[2] / sizeof(src1_t);
        size_t s13 = cnb1[3] / sizeof(src1_t);

        size_t s20 = cnb2[0] / sizeof(src2_t);
        size_t s21 = cnb2[1] / sizeof(src2_t);
        size_t s22 = cnb2[2] / sizeof(src2_t);
        size_t s23 = cnb2[3] / sizeof(src2_t);

        GGML_ASSERT(cnb[0] % sizeof(dst_t) == 0 && cnb[1] % sizeof(dst_t) == 0 && cnb[2] % sizeof(dst_t) == 0 &&
                    cnb[3] % sizeof(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % sizeof(src0_t) == 0 && cnb0[1] % sizeof(src0_t) == 0 && cnb0[2] % sizeof(src0_t) == 0 &&
                    cnb0[3] % sizeof(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % sizeof(src1_t) == 0 && cnb1[1] % sizeof(src1_t) == 0 && cnb1[2] % sizeof(src1_t) == 0 &&
                    cnb1[3] % sizeof(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % sizeof(src2_t) == 0 && cnb2[1] % sizeof(src2_t) == 0 && cnb2[2] % sizeof(src2_t) == 0 &&
                    cnb2[3] % sizeof(src2_t) == 0);

        const src0_t * src0_dd = (const src0_t *) src0->data;
        const src1_t * src1_dd = (const src1_t *) src1->data;
        const src2_t * src2_dd = (const src2_t *) src2->data;
        dst_t *        dst_dd  = (dst_t *) dst->data;

        const int block_size = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> block_dims(1, 1, 1);
        block_dims[2] = std::min<unsigned int>(hne0, block_size);
        block_dims[1] = std::min<unsigned int>(ne1, block_size / (unsigned int) block_dims[2]);
        block_dims[0] = std::min(std::min<unsigned int>(ne2 * ne3,
                                                        block_size / (unsigned int) block_dims[2] /
                                                            (unsigned int) block_dims[1]),
                                 64U);

        sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                  (ne1 + block_dims[1] - 1) / block_dims[1],
                                  (hne0 + block_dims[2] - 1) / block_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        if (block_nums[0] > 65535) {
            int block_num = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) * sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } else {
            stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          item_ct1);
                                 });
        }
    }
}

void ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    const ggml_tensor * src0 = add0->src[0];
    const ggml_tensor * src1 = add0->src[1];
    const ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_add_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_add_kernel_supports(add0->type, src2->type, dst->type));

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_add, float, float, float, float>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, float, float, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, float, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, float, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_add, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_add, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, float, float, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, float,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, float, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#endif
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}
