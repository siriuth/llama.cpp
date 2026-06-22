#ifndef SIRIUTH_WRAPPER_HPP
#define SIRIUTH_WRAPPER_HPP

//#include "siriuth.hpp"
#include "common.hpp"
#include <functional>
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"

#define SYCL_QUEUE_WAIT_TIMMING 32

/*
void ggml_sycl_looper(
        const sycl::range<3>,
        const sycl::range<3>,
        const int,
        queue_ptr,
        std::function<void(sycl::range<3>, sycl::range<3>)>
);
*/

template <typename SubmitFunc>
void ggml_sycl_looper(
        const sycl::range<3> size_range,
        const sycl::range<3> local,
        const int group_size,
        queue_ptr stream,
        SubmitFunc submit_func)
{
    sycl::device dev = stream->get_device();
    const int max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    int group_cnt0 = (size_range[0] + local[0] -1)/local[0];
    int group_cnt1 = (size_range[1] + local[1] -1)/local[1];
    int group_cnt2 = (size_range[2] + local[2] -1)/local[2];

    int wait_cnt = 0;
    int i0 = 0;
    while(i0 < group_cnt0){
        const int global0 = MAX(MIN(group_cnt0-i0, MIN(group_size, max_work_group_size)), 1);
        int i1 = 0;
        while(i1 < group_cnt1){
            const int global1 = MAX(MIN(group_cnt1-i1, MIN(group_size, max_work_group_size)), 1);
            int i2 = 0;
            while(i2 < group_cnt2){
                const int global2 = MAX(MIN(group_cnt2-i2, MIN(group_size, max_work_group_size)), 1);

                if(wait_cnt++%SYCL_QUEUE_WAIT_TIMMING==0){
                    GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
                    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
                }

                //submit_func(sycl::range<3> (global0, global1, global2)*local, sycl::range<3> (i0, i1, 12));
                submit_func(
                    sycl::range<3> (global0, global1, global2) * local,
                    sycl::range<3> (i0, i1, i2) * local
                );

                i2+=global2;

            }
            i1+=global1;
        }
        i0+=global0;
    }
}

#endif
