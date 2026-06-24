#include "fill.hpp"
#include "common.hpp"
#include "siriuth.hpp"

#define SYCL_FILL_WORK_GROUP_NUM 99999
#define SYCL_FILL_WORK_GROUP_SIZE 32
//#define SYCL_FILL_WORK_GROUP_SIZE 64
#define SYCL_FILL_SUB_GROUP_SIZE 16

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
static void fill_kernel_offset(
    T * dst, const int64_t k, const T value,
    const int offset,
    const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t)item.get_global_id(0) + offset;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

/*
template <typename T>
static void fill_kernel(T * dst, const int64_t k, const T value,
                        const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t)item.get_global_id(0);
    if (i >= k) {
        return;
    }
    dst[i] = value;
}
*/

inline void ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    //const int64_t num_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    void * dst_d = dst->data;

    //sycl::device dev = stream->get_device();
    //const int64_t max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
    //GGML_SYCL_DEBUG("[SYCL] %s k:%ld max_work_group_size:%ld\n", __func__, k, max_work_group_size);
    //GGML_ASSERT(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

    switch (dst->type) {
        case GGML_TYPE_F32:
            {
                int world = k;
                int local = SYCL_FILL_WORK_GROUP_SIZE;
                ggml_sycl_looper(world, local, SYCL_FILL_WORK_GROUP_NUM, stream,
                    //[=](sycl::range<1> global, sycl::range<1> offset){
                    [=](int global, int offset){

                    stream->parallel_for(
                        sycl::nd_range<1>(global, local),
                        [=](sycl::nd_item<1> item)
                        [[sycl::reqd_sub_group_size(SYCL_FILL_SUB_GROUP_SIZE)]]
                        {
                            fill_kernel_offset(
                                static_cast<float *>(dst_d), k, value,
                                offset, item);
                        });

                    });

            }
            break;
        case GGML_TYPE_F16:
            {
                sycl::half h_value = sycl::half(value);
                int world = k;
                int local = SYCL_FILL_WORK_GROUP_SIZE;
                ggml_sycl_looper(world, local, SYCL_FILL_WORK_GROUP_NUM, stream,
                    [=](int global, int offset){
                    //[=]sycl::range<1> global, sycl::range<1> offset){

                        stream->parallel_for(sycl::nd_range<1>(global, local),
                            [=](sycl::nd_item<1> item)
                            [[sycl::reqd_sub_group_size(SYCL_FILL_SUB_GROUP_SIZE)]]
                            {
                                fill_kernel_offset(
                                    static_cast<sycl::half *>(dst_d), k, h_value,
                                    offset, item);
                            });
                    });
            }
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

void ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
