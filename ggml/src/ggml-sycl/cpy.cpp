#include "cpy.hpp"

#include <float.h>

#include "siriuth.hpp"
#include "dequantize.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"

#define SYCL_CPY_WORK_GROUP_NUM 99999
#define SYCL_CPY_WORK_GROUP_SIZE 32
#define SYCL_CPY_SUB_GROUP_SIZE 16

static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    float *       dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<float, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    float *            dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i16_i16(const char * cxi, char * cdsti) {
    const int16_t * xi   = (const int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16_op_same_shape(const char * cx, char * cdst, const int ne,
                        const int ne00, const int ne01, const int ne02, const int ne03,
                        const int nb00, const int nb01, const int nb02, const int nb03,
                        //const int ne10, const int ne11, const int ne12, const int ne13,
                        const int nb10, const int nb11, const int nb12, const int nb13,
                        const int offset0, const int offset1, const int offset2,
                        const sycl::nd_item<3> & item_ct1) {
    // 要素数が同じ、形状も同じ物
    const int i0 = item_ct1.get_global_id(2) + offset2;
    const int i1 = item_ct1.get_global_id(1) + offset1;
    const int i2 = (item_ct1.get_global_id(0) + offset0) / ne03;
    const int i3 = (item_ct1.get_global_id(0) + offset0) % ne03;

    if (i0 >= ne00 || i1 >= ne01 || i2 >= ne02) {
        return;
    }

    const int i_src0 =  i3*nb03 +  i2*nb02 +  i1*nb01 +  i0*nb00;
    const int i_src1 =  i3*nb13 +  i2*nb12 +  i1*nb11 +  i0*nb10;

    cpy_1(cx + i_src0, cdst + i_src1);
}

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16_op_different_shape(const char * cx, char * cdst, const int ne,
                        const int ne00, const int ne01, const int ne02, const int ne03,
                        const int nb00, const int nb01, const int nb02, const int nb03,
                        const int ne10, const int ne11, const int ne12, const int ne13,
                        const int nb10, const int nb11, const int nb12, const int nb13,
                        const int offset0, const int offset1, const int offset2,
                        const sycl::nd_item<3> & item_ct1) {
    // 要素数が同じだが、次元ごとの要素数（形状）は違うもの（あり得るのかどうかわからない）
    const int i0 = item_ct1.get_global_id(2) + offset2;
    const int i1 = item_ct1.get_global_id(1) + offset1;
    const int i2 = (item_ct1.get_global_id(0) + offset0) / ne03;
    const int i3 = (item_ct1.get_global_id(0) + offset0) % ne03;

    if (i0 >= ne00 || i1 >= ne01 || i2 >= ne02) {
        return;
    }

    //if (i >= ne) {
    //    return;
    //}
    // 一旦一次元に変換
    const int i = i0 + i1 * ne10 + i2 * ne10 * ne11 + i3 * ne10 * ne11 * ne12;
    // 形状に合わせる
    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;

    const int i_src0 =  i3*nb03 +  i2*nb02 +  i1*nb01 +  i0*nb00;
    const int i_src1 = i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10;

    cpy_1(cx + i_src0, cdst + i_src1);
}

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                        const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                        const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
static void cpy_blck_q_q(const char * cxi, char * cdsti) {
    const T * xi = (const T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


static void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}



template <dequantize_kernel_t dequant, int qk> static void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, int qk>
static void cpy_q_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }


    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_q_f32(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_sycl(
    const char * cx, char * cdst, const int ne  ,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12, const int ne13,
    const int nb10, const int nb11, const int nb12, const int nb13,
    queue_ptr stream)
{
/*
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        GGML_SYCL_DEBUG("[SYCL] %s SYCL_CPY_BLOCK_SIZE:%d SYCL_MAX_WORK_GROUP_SIZE:%d SYCL_CPY_SUBGROUP_SIZE:%d\n", __func__
                     , SYCL_CPY_BLOCK_SIZE, SYCL_MAX_WORK_GROUP_SIZE, SYCL_CPY_SUBGROUP_SIZE);
        if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12)){
//        if(false){
            // 前提条件下限定の処理
            // 前提条件 : ne00～02 と ne11～12が同じ
            // 1次元で扱い、最適化する
            int i = 0;
            while(i < ne){
                    stream->parallel_for(
                        sycl::nd_range<1>(
                            sycl::range<1>(MIN(ne - i, SYCL_CPY_WORK_ITEM_SIZE * SYCL_MAX_WORK_GROUP_SIZE)), sycl::range<1>(MIN(ne - i, SYCL_CPY_WORK_ITEM_SIZE)))
                          , [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUBGROUP_SIZE)]] {
                            cpy_f32_f16_opt<cpy_1_f16_f32>(
                            cx, cdst, ne,
                            //ne00, ne01, ne02,
                            nb00, //nb01, nb02, nb03,
                            //ne10, ne11, ne12,
                            nb10,//nb11, nb12, nb13,
                            i, //src_base, dst_base,
                            item_ct1
                            );
                        }
                    );
                 i += MIN(ne - i, SYCL_CPY_WORK_ITEM_SIZE * SYCL_MAX_WORK_GROUP_SIZE);
            }
        } else {
            // 基本となる実装されていた処理
            const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
            const int ne000102 = ne00 * ne01 * ne02;
            const int ne101112 = ne10 * ne11 * ne12;
            stream->parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE))
                  , [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUBGROUP_SIZE)]] {
                    cpy_f32_f16<cpy_1_f16_f32>(cx, cdst
                      , ne  , ne00, ne01, ne02
                      , nb00, nb01, nb02, nb03
                            , ne10, ne11, ne12
                      , nb10, nb11, nb12, nb13
                      , ne000102, ne101112
                      , item_ct1
                    );
                }
            );
        }
*/
/*
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    sycl::device dev = stream->get_device();
    const int max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    const sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    int group_cnt0 = (ne02*ne03 + local[0] -1)/local[0];
    int group_cnt1 = (ne01 + local[1] -1)/local[1];
    int group_cnt2 = (ne00 + local[2] -1)/local[2];

    int wait_cnt = 0;
    int i0 = 0;
    while(i0 < group_cnt0){
        const int global0 = MAX(MIN(group_cnt0-i0, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);
        int i1 = 0;
        while(i1 < group_cnt1){
            const int global1 = MAX(MIN(group_cnt1-i1, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);
            int i2 = 0;
            while(i2 < group_cnt2){
                const int global2 = MAX(MIN(group_cnt2-i2, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);

                if(wait_cnt++%SYCL_QUEUE_WAIT_TIMMING==0){
                    GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
                    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
                }

                if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12) && (ne03 == ne13)){
                    stream->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(global0, global1, global2) * local, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_same_shape<cpy_1_f16_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                //ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                i0 *local[0], i1 * local[1], i2 * local[2],
                                item_ct1
                            );
                        }
                    );
                } else {
                    stream->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(global0, global1, global2) * local, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f16_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                i0 *local[0], i1 * local[1], i2 * local[2],
                                item_ct1
                            );
                        }
                    );
                }

                i2+=global2;
            }
            i1+=global1;
        }
        i0+=global0;
    }
*/
/*
    sycl::range<3> world(ne03*ne02, ne01, ne00);
    sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    //sycl::range<3> offset(0, 0, 0);

    ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
        [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                            sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f16_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                //offset[0] * local[0], offset[1] * local[1], offset[2] * local[2],
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );
        });
*/


    sycl::range<3> world(ne03*ne02, ne01, ne00);
    sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12) && (ne03 == ne13)){
        GGML_SYCL_DEBUG("[SYCL] %s same shape\n", __func__);
        ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_same_shape<cpy_1_f16_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                //ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );

            });
    } else {
        GGML_SYCL_DEBUG("[SYCL] %s different shape\n", __func__);
        ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f16_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );

            });


    }

}

static void ggml_cpy_f32_f32_sycl(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12, const int ne13,
    const int nb10, const int nb11, const int nb12, const int nb13,
    queue_ptr stream)
{
/*
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    sycl::device dev = stream->get_device();
    const int max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    const sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    int group_cnt0 = (ne02*ne03 + local[0] -1)/local[0];
    int group_cnt1 = (ne01 + local[1] -1)/local[1];
    int group_cnt2 = (ne00 + local[2] -1)/local[2];

    int wait_cnt = 0;
    int i0 = 0;
    while(i0 < group_cnt0){
        const int global0 = MAX(MIN(group_cnt0-i0, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);
        int i1 = 0;
        while(i1 < group_cnt1){
            const int global1 = MAX(MIN(group_cnt1-i1, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);
            int i2 = 0;
            while(i2 < group_cnt2){
                const int global2 = MAX(MIN(group_cnt2-i2, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);

                if(wait_cnt++%SYCL_QUEUE_WAIT_TIMMING==0){
                    GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
                    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
                }

                if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12) && (ne03 == ne13)){
                    // 同じ形状の物（これ以外対応する必要はないと思うのだが…わからん）
                    stream->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(global0, global1, global2) * local, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_same_shape<cpy_1_f32_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                //ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                i0 *local[0], i1 * local[1], i2 * local[2],
                                item_ct1
                            );
                        }
                    );
                } else {
                    // 形状が違うもの 線形にして処理するのが正しいのか、範囲で切り取るのか、どちらが正しいかわからん。
                    // 線形で処理してコピー先の形状に合わせてすべての要素数を複写する形にしておくか…
                    stream->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(global0, global1, global2) * local, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f32_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                i0 *local[0], i1 * local[1], i2 * local[2],
                                item_ct1
                            );
                        }
                    );
                }

                i2+=global2;
            }
            i1+=global1;
        }
        i0+=global0;
    }
*/
    sycl::range<3> world(ne03*ne02, ne01, ne00);
    sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12) && (ne03 == ne13)){
        GGML_SYCL_DEBUG("[SYCL] %s same shape\n", __func__);
        ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_same_shape<cpy_1_f32_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                //ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );

            });
    } else {
        GGML_SYCL_DEBUG("[SYCL] %s different shape\n", __func__);
        ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f32_f32>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );

            });


    }

/*
    } else {
        const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
        const int ne000102 = ne00 * ne01 * ne02;
        const int ne101112 = ne10 * ne11 * ne12;

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                      sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(SYCL_CPY_SUBGROUP_SIZE)]]
            {
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                   nb10, nb11, nb12, nb13,
                                   ne000102, ne101112,
                                   item_ct1);
            }
        );
    }
*/
}

static void ggml_cpy_f32_f16_sycl(
const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int nb00, const int nb01, const int nb02, const int nb03,
    const int ne10, const int ne11, const int ne12, const int ne13,
    const int nb10, const int nb11, const int nb12, const int nb13,
    queue_ptr stream) {

/*
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        //GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
        if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12)){
            int i = 0;
            while(i < ne){
                stream->parallel_for(
                    sycl::nd_range<1>(
                        sycl::range<1>(MIN(ne - i, SYCL_CPY_WORK_ITEM_SIZE * SYCL_MAX_WORK_GROUP_SIZE)),
                        sycl::range<1>(MIN(ne - i, SYCL_CPY_WORK_ITEM_SIZE)) ),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUBGROUP_SIZE)]] {
                        cpy_f32_f16_opt<cpy_1_f32_f16>(
                            cx, cdst, ne,
                            //ne00, ne01, ne02,
                            nb00, //nb01, nb02, nb03,
                            //ne10, ne11, ne12,
                            nb10,//nb11, nb12, nb13,
                            i, //src_base, dst_base,
                            item_ct1
                        );
                    }
                );
                i += MIN(ne - i, SYCL_CPY_WORK_ITEM_SIZE * SYCL_MAX_WORK_GROUP_SIZE);
            }
        } else {
            const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
            const int ne000102 = ne00 * ne01 * ne02;
            const int ne101112 = ne10 * ne11 * ne12;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(SYCL_CPY_SUBGROUP_SIZE)]]
                {
                    cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13
                                         , ne000102, ne101112
                                         , item_ct1);
                });
        }
    }
*/
/*
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    sycl::device dev = stream->get_device();
    const int max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    const sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    int group_cnt0 = (ne02*ne03 + local[0] -1)/local[0];
    int group_cnt1 = (ne01 + local[1] -1)/local[1];
    int group_cnt2 = (ne00 + local[2] -1)/local[2];

    int wait_cnt = 0;
    int i0 = 0;
    while(i0 < group_cnt0){
        const int global0 = MAX(MIN(group_cnt0-i0, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);
        int i1 = 0;
        while(i1 < group_cnt1){
            const int global1 = MAX(MIN(group_cnt1-i1, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);
            int i2 = 0;
            while(i2 < group_cnt2){
                const int global2 = MAX(MIN(group_cnt2-i2, MIN(SYCL_CPY_WORK_GROUP_NUM, max_work_group_size)), 1);

                if(wait_cnt++%SYCL_QUEUE_WAIT_TIMMING==0){
                    GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
                    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
                }

                if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12) && (ne03 == ne13)){
                    stream->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(global0, global1, global2) * local, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_same_shape<cpy_1_f32_f16>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                //ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                i0 *local[0], i1 * local[1], i2 * local[2],
                                item_ct1
                            );
                        }
                    );
                } else {
                    stream->parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(global0, global1, global2) * local, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f32_f16>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                i0 *local[0], i1 * local[1], i2 * local[2],
                                item_ct1
                            );
                        }
                    );
                }

                i2+=global2;
            }
            i1+=global1;
        }
        i0+=global0;
    }
*/

    sycl::range<3> world(ne03*ne02, ne01, ne00);
    sycl::range<3> local(1, 1, SYCL_CPY_WORK_GROUP_SIZE);
    if((ne00 > 0) && (ne00 == ne10) && (ne01 == ne11) && (ne02 == ne12) && (ne03 == ne13)){
        GGML_SYCL_DEBUG("[SYCL] %s same shape\n", __func__);
        ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_same_shape<cpy_1_f32_f16>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                //ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );

            });
    } else {
        GGML_SYCL_DEBUG("[SYCL] %s different shape\n", __func__);
        ggml_sycl_looper(world, local, SYCL_CPY_WORK_GROUP_NUM, stream,
            [=](sycl::range<3> global, sycl::range<3> offset){

                    stream->parallel_for(
                        sycl::nd_range<3>(global, local),
                            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(SYCL_CPY_SUB_GROUP_SIZE)]] {
                            cpy_f32_f16_op_different_shape<cpy_1_f32_f16>(
                                cx, cdst, ne,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                offset[0], offset[1], offset[2],
                                item_ct1
                            );
                        }
                    );

            });


    }

}

static void ggml_cpy_f32_q8_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q8_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f32_q4_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q4_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_iq4_nl_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                     const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                     const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                     const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f16_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i16_i16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i32_i32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_q8_0_q8_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_0_q5_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_1_q5_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_0_q4_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_1_q4_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {

   const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
   stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) try {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_TENSOR_BINARY_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;
    if ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
