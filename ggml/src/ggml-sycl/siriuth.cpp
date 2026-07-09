#include "siriuth.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include "common.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"

 // μ秒 1sec. = 1000000 micro sec.
#define SLEEP_TIME 200
//#define SLEEP_TIME 500
//#define SLEEP_TIME 1000
//#define SLEEP_TIME 10
//#define SLEEP_TIME_MAX 100000
#define SLEEP_TIME_MAX 999999


SyclQueueEventWatcher::SyclQueueEventWatcher()
// : xxx(xxxx)
{
    events.reserve(SYCL_EVENTS_SIZE);
}

SyclQueueEventWatcher& SyclQueueEventWatcher::getInstance()
{
    static SyclQueueEventWatcher instance;
    return instance;
}

int SyclQueueEventWatcher::get_events_size()
{
/*
    int result = 0;
    for(int i = 0; i < SYCL_EVENTS_SIZE; i++){
        if(events[i] != null){
        //if(!events[i].empty()){
            result++;
        }
    }
    return result;
*/
    return events.size();
}

void SyclQueueEventWatcher::SetEvent(sycl::event e)
{
    events.push_back(e);
/*
    for(int i = 0; i < SYCL_EVENTS_SIZE; i++){
        //if(events[i] == null){
        if(events[i].empty()){
            events[i] = e;
            return;
        }
    }
    GGML_ASSERT(false); // オバーフロー
*/
}

void SyclQueueEventWatcher::WaitForSubmit()
{
    // 状態を出力するための変数（出力しなければ不要）
    //const size_t event_size = events.size();
    const int64_t t0 = ggml_time_ms();
    const size_t event_size = get_events_size();
    size_t event_submitted_cnt = 0;
    size_t event_running_cnt = 0;
    size_t event_complete_cnt = 0;
    size_t sleep_cnt = 0;
    // 処理上必要な変数
    size_t chk_cnt = 0;
    //size_t e_cnt = events.size();
    size_t e_cnt = get_events_size();
    //size_t e_cnt = event_size;
    size_t sleep_time = SLEEP_TIME;
    while(e_cnt > SYCL_QUEUE_WAIT_SIZE){
        event_submitted_cnt = 0;
        event_running_cnt = 0;
        for(auto it = events.begin(); it != events.end();){
        //for(int i = 0; i < SYCL_EVENTS_SIZE; i++){
        //for(const auto& it : events){
            //auto status = it->get_info<sycl::info::event_command_status>();
            auto status = it->get_info<sycl::info::event::command_execution_status>();
            //auto status = it->get_info<sycl::info::event_profiling>();
            if (status == sycl::info::event_command_status::complete) {
                events.erase(it);
                event_complete_cnt++;
                //GGML_SYCL_DEBUG("[SYCL] %s event erase.\n", __func__);
            }else{
                if (status == sycl::info::event_command_status::submitted){
                    event_submitted_cnt++;
                }else if (status == sycl::info::event_command_status::running){
                    event_running_cnt++;
                }
                it++;
            }
/*
            // コンテナの要素を空にする方法が分からないので実装不可
            //if(events[i] != null){
            if(!events[i].empty()){
                auto status = events[i].get_info<sycl::info::event::command_execution_status>();
                if (status == sycl::info::event_command_status::complete) {
                    // イベントオブジェクトの解放をしないとダメ？
                    events[i] = empty();
                    event_complete_cnt++;
                    //GGML_SYCL_DEBUG("[SYCL] %s event erase.\n", __func__);
                }else{
                    if (status == sycl::info::event_command_status::submitted){
                        event_submitted_cnt++;
                    }else if (status == sycl::info::event_command_status::running){
                        event_running_cnt++;
                    }
                }
            }
*/
        }
        //e_cnt = events.size();
        e_cnt = get_events_size();
        //if(events.size() == s){
        if(e_cnt > SYCL_QUEUE_WAIT_SIZE){
            if(chk_cnt > 1000){
                GGML_SYCL_DEBUG("[SYCL] %s chk_cnt over 1000.\n", __func__);
                chk_cnt = 0;
            }
            // 負荷を減らすために少し待つ
            //GGML_SYCL_DEBUG("[SYCL] %s %zumicro sec. sleep for waiting. events:%zu\n", __func__, sleep_time, e_cnt);
            //std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
            std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_TIME));
            sleep_cnt++;
            sleep_time = MIN(sleep_time * 2, SLEEP_TIME_MAX);
            chk_cnt++;
        }
    }
    //if(event_submitted_cnt!=0||event_running_cnt!=0||event_complete_cnt!=0){
    if(sleep_cnt!=0){
        const int64_t t1 = ggml_time_ms();
        GGML_SYCL_DEBUG("[SYCL] %s %ldms %zu sleep events:%ld -> %ld  complete:%ld submitted:%ld running:%ld\n", __func__,
             (t1 - t0),
             //(t1 - t0) * 1.0f / 1000 // %.2fs,
             sleep_cnt,
             event_size, e_cnt, event_complete_cnt, event_submitted_cnt, event_running_cnt
        );
    }

/*
    size_t completed_count = 0;
        size_t current_completed = 0;
        for (const sycl::event& e : events) {
            //sycl::info::event::command_execution_status
            //auto status = e.template get_info<sycl::info::event::event_command_status>();
            auto status = e.template get_info<sycl::info::event::command_execution_status>();
            if (status == sycl::info::event_command_status::complete) {
                current_completed++;
            }

        }
*/
}
