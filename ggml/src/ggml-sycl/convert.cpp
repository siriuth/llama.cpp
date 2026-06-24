#include "convert.hpp"
#include "dequantize.hpp"
#include "presets.hpp"
#include "siriuth.hpp"


#define SYCL_DEQUANTIZE_WORK_GROUP_NUM 99999
#define SYCL_DEQUANTIZE_WORK_GROUP_SIZE 32
#define SYCL_DEQUANTIZE_SUB_GROUP_SIZE 16

//#define MAX_GRIDDIM_Y 65535
//#define MAX_GRIDDIM_Z 65535
#define SYCL_UNARY_WORK_GROUP_NUM 99999
//#define MAX_GRIDDIM_Y 512
//#define MAX_GRIDDIM_Z 512
//#define MAX_WORK_GROUP_SIZE 512
#define SYCL_UNARY_WORK_GROUP_SIZE 32
#define SYCL_UNARY_SUB_GROUP_SIZE 16
//#define SYCL_UNARRAY_BLOCK_SIZE 8   //  36.39s   662
//#define SYCL_UNARRAY_BLOCK_SIZE 16  //  29.98s   657 vae best
//#define SYCL_UNARRAY_BLOCK_SIZE 32  //  30.71s   655
//#define SYCL_UNARRAY_BLOCK_SIZE 64  //  36.12s   653 total best
//#define SYCL_UNARRAY_BLOCK_SIZE 128 //  50.88s   667
//#define SYCL_UNARRAY_BLOCK_SIZE 256 //  81.16s   697
//#define SYCL_UNARRAY_BLOCK_SIZE 512 // 142.24s   762

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k,
                             const sycl::nd_item<3> &item_ct1) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void *__restrict__ vx,
                                  dst_t *__restrict__ y, const int64_t k,
                                  dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t num_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                             dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y, item_ct1, nb);
        });
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    int constexpr WARP_K = WARP_SIZE * QK4_0;
    const int n_warp = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % 2 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) *
        sycl::range<3>(1, 1, WARP_SIZE),
        sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q8_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    int constexpr WARP_K = WARP_SIZE * QK8_0;
    const int n_warp = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % QK8_0 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) *
        sycl::range<3>(1, 1, WARP_SIZE),
        sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            dequantize_block_q8_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q4_1_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_K(vx, y, get_pointer(scale_local_acc), item_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    const size_t  local_size  = 32;
    const size_t  global_size = nb * local_size;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y, get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(K_SCALE_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block_q5_K_reorder(vx, y, get_pointer(scale_local_acc), item_ct1, nb);
            });
    });
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y, item_ct1, nb); });
}

template <typename dst_t>
static void dequantize_row_iq1_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                      dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


template <typename dst_t>
static void dequantize_row_iq3_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#else
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#endif
}

template <typename dst_t>
static void dequantize_row_iq4_nl_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

template <typename dst_t>
static void dequantize_row_mxfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    const int nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename dst_t>
static void dequantize_row_nvfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}

/*
template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #else
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}

*/

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc_offset(const void * __restrict__ vx, dst_t * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03,
    const sycl::range<3> offset,
    const sycl::nd_item<3> & item_ct1)
{
    //auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    //const int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));
    const int64_t i00 = (item_ct1.get_global_id(2) + offset[2]) * 2;
    const int64_t i01 = item_ct1.get_group(1) + offset[1];
    const int64_t i02s= item_ct1.get_group(0) + offset[0];
    const int64_t i02 = i02s / ne03;

    if (i00 >= ne00 || i01 >= ne01 || i02s >= ne02*ne03 || i02 >= ne02) {
        return;
    }
    const int64_t i03 = i02s % ne03;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #else
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc_sycl(const void *    vx,
                                  dst_t *         y,
                                  const int64_t   ne00,
                                  const int64_t   ne01,
                                  const int64_t   ne02,
                                  const int64_t   ne03,
                                  const int64_t   s01,
                                  const int64_t   s02,
                                  const int64_t   s03,
                                  dpct::queue_ptr stream) {

/*
    const dpct::dim3 num_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(num_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
*/

    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    sycl::range<3> world(ne03*ne02, ne01, ne00 / 2);
    sycl::range<3> local(1, 1, SYCL_DEQUANTIZE_WORK_GROUP_SIZE);
    ggml_sycl_looper(world, local, SYCL_DEQUANTIZE_WORK_GROUP_NUM, stream,
        [=](sycl::range<3> global, sycl::range<3> offset){

            stream->parallel_for(
                sycl::nd_range<3>(global, local),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(SYCL_DEQUANTIZE_SUB_GROUP_SIZE)]] {
                    dequantize_block_nc_offset<qk, qr, dequantize_kernel>(
                        vx, y,
                        ne00, ne01, ne02, ne03,
                        s01, s02, s03,
                        offset, item_ct1);
                });


        }
    );

}

template <typename src_t, typename dst_t>
static void convert_unary_nc_one_offset(
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const int offset,
    const sycl::nd_item<1> & item_ct1) {

    const size_t i = item_ct1.get_global_id(0) + offset;
    const src_t * x = static_cast<const src_t *>(vx);
    y[i] = static_cast<dst_t>(x[i]);
}

/*
template <typename src_t, typename dst_t>
static void convert_unary_nc_one(const void * __restrict__ vx, dst_t * __restrict__ y,
                          const sycl::nd_item<1> & item_ct1) {

    const size_t offset = item_ct1.get_local_range(0) * item_ct1.get_group(0)
                        + item_ct1.get_local_id(0);
    const src_t * x = static_cast<const src_t *>(vx);
    y[offset] = static_cast<dst_t>(x[offset]);
}
*/

/*
template <typename src_t, typename dst_t>
static void convert_unary_nc_block(const void * __restrict__ vx, dst_t * __restrict__ y,
//                          const int64_t ne00, const int64_t ne01,
//                          const size_t offset_base, // item_ct1.get_local_range(0) * item_ct1.get_group(0)
                          const sycl::nd_item<1> & item_ct1) {

//    32 x 0..512
    const size_t offset = item_ct1.get_local_range(0) * item_ct1.get_group(0)
                        + item_ct1.get_local_id(0); // まちがってね？ｗ
// 0..512 x local_id 0..32
//    const size_t offset = offset_base + item_ct1.get_local_id(0);

    const src_t * x = static_cast<const src_t *>(vx);
#pragma unroll
    for (size_t i = 0; i < SYCL_UNARRAY_BLOCK_SIZE; i++) {
        const size_t ix = offset + i;
        //const size_t ix = item_ct1.get_local_id(0) + i;
        //item_ct1.barrier();
        y[ix] = static_cast<dst_t>(x[ix]);
    }
}
*/

/*
template <typename src_t, typename dst_t>
static void convert_unary_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
                          const int64_t ne00, const int64_t ne01,
                          const sycl::nd_item<1> & item_ct1) {

    const size_t offset = item_ct1.get_local_range(0) * item_ct1.get_group(0);
    const int64_t i00 = offset + item_ct1.get_local_id(0);

    //if (i00 >= ne00) {
    //    return;
    //}

    const src_t * x = static_cast<const src_t *>(vx);
    // vae 512x512 96.11s
    const size_t max = MIN(item_ct1.get_local_range(0), (size_t)(ne00 - i00));
    for (size_t i = item_ct1.get_local_id(0); i < max; i++) {
        //const int64_t ix = item_ct1.get_local_range(0) * item_ct1.get_group(0) + i;
        const size_t ix = offset + i;
        //const int64_t iy = ix;
        //item_ct1.barrier();
        y[ix] = static_cast<dst_t>(x[ix]);
    }
}
*/

template <typename src_t, typename dst_t>
static void convert_unary_nc_sycl(const void * __restrict__ vx, dst_t * __restrict__ y,
    // 引数の変数名は完全にデタラメ
    //const int64_t ne00, const int64_t ne01,const int64_t ne02, const int64_t ne03,
    const int64_t nc, const int64_t ne01,const int64_t ne02, const int64_t ne03,
    const int64_t s01, const int64_t s02, const int64_t s03,
    dpct::queue_ptr queue) {
    GGML_SYCL_DEBUG("[SYCL] %s\n", __func__);
    // 現状ではconvert_unary_syclから呼び出されることしかないので、
    // メモリ配列の長さのみ参照して連続領域に書き込む処理しかない。
    // 今後まともな実装を行う必要が出てくる可能性はある。
    GGML_ASSERT(queue);
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    GGML_UNUSED(ne01);
    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);
    GGML_UNUSED(s01);
    GGML_UNUSED(s02);
    GGML_UNUSED(s03);
    //GGML_SYCL_DEBUG("[SYCL] %s ne02_fdv(%u,%u,%u)\n", __func__, ne02_fdv[0], ne02_fdv[1], ne02_fdv[2]);

    const src_t * x = static_cast<const src_t *>(vx);

    //GGML_SYCL_DEBUG("[SYCL] %s nc:%ld\n", __func__, nc);

    int world = nc;
    int local = SYCL_UNARY_WORK_GROUP_SIZE;
    ggml_sycl_adjusted_looper(world, local, SYCL_UNARY_WORK_GROUP_NUM, queue,
        [=](int adjusted_global, int adjusted_local, int offset){

        queue->parallel_for(sycl::nd_range<1>(adjusted_global, adjusted_local),
                    [=](sycl::nd_item<1> item_ct1)
                    [[sycl::reqd_sub_group_size(SYCL_UNARY_SUB_GROUP_SIZE)]] {
                        convert_unary_nc_one_offset<src_t>(
                            (const src_t *)(x),
                            (dst_t *)(y),
                            offset,
                            item_ct1
                        );
                    }
        );
}

template <typename src_t, typename dst_t>
static void convert_unary_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr queue) {
    //GGML_SYCL_DEBUG("[SYCL] %s k:%ld\n", __func__, k);
    //int64_t local_x = MIN(k, SYCL_UNARRAY_BLOCK_SIZE);
    //int64_t global_x = k;
    //int64_t global_x = MIN(k, SYCL_UNARRAY_BLOCK_SIZE); // この値使われていないけど意味合い間違ってるｗ
    //const sycl::device dev = queue->get_device();
    //const int32_t max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    //const size_t max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
    //GGML_SYCL_DEBUG("[SYCL] %s max_compute_units:%d max_work_group_size:%zu)\n", __func__, max_compute_units, max_work_group_size);

    //GGML_SYCL_DEBUG("[SYCL] %s local x:%ld\n", __func__, local_x);
    //GGML_SYCL_DEBUG("[SYCL] %s global x:%ld\n", __func__, global_x);

    //convert_unary_nc_sycl<src_t>(vx, y, k, local_x, 1, 1, global_x, max_compute_units, max_work_group_size, queue);
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, 1, 1, 1, queue);
    //convert_unary_nc_sycl<src_t>(vx, y, k, local_x, global_x, queue);
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q8_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
            }
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q3_K_sycl_reorder;
            } else {
                return dequantize_row_q3_K_sycl;
            }
        case GGML_TYPE_Q4_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q5_K_sycl_reorder;
            } else {
                return dequantize_row_q5_K_sycl;
            }
        case GGML_TYPE_Q6_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_sycl;
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_row_q4_0_sycl;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_sycl;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q8_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
            }
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q3_K_sycl_reorder;
            } else {
                return dequantize_row_q3_K_sycl;
            }
        case GGML_TYPE_Q4_K:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q5_K_sycl_reorder;
            } else {
                return dequantize_row_q5_K_sycl;
            }
        case GGML_TYPE_Q6_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_sycl;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_nc_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        case GGML_TYPE_Q1_0:
            return dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        default:
            return nullptr;
    }
}
