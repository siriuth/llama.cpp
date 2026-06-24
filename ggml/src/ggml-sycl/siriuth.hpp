#ifndef SIRIUTH_WRAPPER_HPP
#define SIRIUTH_WRAPPER_HPP

// 2026/06/23 wait()表示をコメントアウト

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
                    //GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
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

template <typename SubmitFunc>
void ggml_sycl_looper(
        const int size_range,
        const int local,
        const int group_size,
        queue_ptr stream,
        SubmitFunc submit_func)
{
    sycl::device dev = stream->get_device();
    const int max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    int group_cnt = (size_range + local -1)/local;

    int wait_cnt = 0;
    int i = 0;
    while(i < group_cnt){
        const int global = MAX(MIN(group_cnt-i, MIN(group_size, max_work_group_size)), 1);

        if(wait_cnt++%SYCL_QUEUE_WAIT_TIMMING==0){
            //GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
            SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
        }

        submit_func(
            (global * local),
            (i * local)
        );

        i+=global;
    }
}

// 範囲外に出さない処理
template <typename SubmitFunc>
void ggml_sycl_adjusted_looper(
        const int size_range,
        const int local,
        const int group_size,
        queue_ptr stream,
        SubmitFunc submit_func)
{
    sycl::device dev = stream->get_device();
    const int max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    int group_cnt = (size_range + local -1)/local;

    int adjusted_group_quantity = 0;
    int wait_cnt = 0;
    int i = 0;
    //GGML_SYCL_DEBUG("[SYCL] %s size_range:%d local:%d group_size:%d group_cnt:%d\n", __func__, size_range, local, group_size, group_cnt);
    while(i < group_cnt){
        adjusted_group_quantity = MAX(MIN(group_cnt-i, MIN(group_size, max_work_group_size)), 1);
        //GGML_SYCL_DEBUG("[SYCL] %s i:%d adjusted_group_quantity:%d\n", __func__, i, adjusted_group_quantity);

        // 残りの数 ＞＝ 今回の数 普通の処理
        // 残りの数 ＜ 今回の数 端数処理へ 端数処理ではlocalのブロックを一括で出力して最後の一つをグローバルとローカルを揃えて出力
        //if(size_range-(group_cnt-i)*local < adjusted_group_quantity * local){
        if(size_range-i*local < adjusted_group_quantity * local){
            break;
        }

        if(wait_cnt++%SYCL_QUEUE_WAIT_TIMMING==0){
            //GGML_SYCL_DEBUG("[SYCL] %s wait()\n", __func__);
            SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
        }

        submit_func(
            (adjusted_group_quantity * local),
            local,
            (i * local)
        );

        i += adjusted_group_quantity;
    }
    // 端数処理
    if(i < group_cnt){
        // 端数処理
        if(size_range - i * local - adjusted_group_quantity * local == 0){
            GGML_SYCL_DEBUG("[SYCL] %s i:%d adjusted_group_quantity:%d local:%d\n", __func__, i, adjusted_group_quantity, local);
            submit_func(
                (adjusted_group_quantity * local),
                local,
                (i * local)
            );
            i += adjusted_group_quantity;
        } else {
           // local単位で処理を行う。
           if(adjusted_group_quantity > 1){
                const int re_adjusted_group_quantity = adjusted_group_quantity - 1;
                GGML_SYCL_DEBUG("[SYCL] %s i:%d re_adjusted_group_quantity:%d local:%d\n", __func__, i, re_adjusted_group_quantity, local);
                submit_func(
                    (re_adjusted_group_quantity * local),
                    local,
                    (i * local)
                );
                i += re_adjusted_group_quantity;
            }
            // localで割り切れなかった余りの処理
            const int adjusted_local = MAX(MIN(size_range - (group_cnt-i)*local, local), 1);
            GGML_SYCL_DEBUG("[SYCL] %s i:%d adjusted_local:%d local:%d\n", __func__, i, adjusted_local, local);
            submit_func(
                    adjusted_local,
                    adjusted_local,
                    (i * local)
            );
            i += 1;
        }
    }
}

#endif
