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
            //sleep_time = MIN(sleep_time * 2, SLEEP_TIME_MAX);
            sleep_time = MIN(sleep_time * 1.2, SLEEP_TIME_MAX);
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

// 要素数が多いもの順で並べる
//void sortDim(int* d, const int64_t ne[4]){
void ggml_sortDim(int* d, const int64_t ne[GGML_MAX_DIMS]){
    //int d[4] = {0, 1, 2, 3}
    //for(int i = 0; i < (4 - 1); i++){
    for(int i = 0; i < (GGML_MAX_DIMS - 1); i++){
        //for(int j = i+1; j < 4; j++){
        for(int j = i+1; j < GGML_MAX_DIMS; j++){
            if(ne[d[i]] < ne[d[j]]){
                std::swap(d[i], d[j]);
            }
        }
    }
    //return d;
}

// データ間隔
// 要素数が１以外でデータの間隔が短いもの
// 要素数が1のものは問答無用で後ろへ
//void sortDim(int* d, const int64_t ne[4], const size_t nb[4]){
void ggml_sortDim(int* d, const int64_t ne[GGML_MAX_DIMS], const size_t nb[GGML_MAX_DIMS]){
    //int d[4] = {0, 1, 2, 3}
    //for(int i = 0; i < (4 - 1); i++){
    for(int i = 0; i < (GGML_MAX_DIMS - 1); i++){
        //for(int j = i+1; j < 4; j++){
        for(int j = i+1; j < GGML_MAX_DIMS; j++){
            if((ne[d[i]] == 1 && ne[d[j]] != 1) || (nb[d[i]] > nb[d[j]])){
                std::swap(d[i], d[j]);
            }
        }
    }
    //return d;
}

// Effective number of dimensions 有効な次元数
// ne配列を次元の末尾から参照して1以外の次元数を返す
// 例)
// ne(4, 1024, 1, 1) -> 2
// ne(1, 1, 123, 1) -> 3 --> sorDimで ne(123, 1, 1, 1)へ変形 -> 1となる。
// sortDimしたあとに使用することを想定。4次元配列未満（3次元配列以下）ならrage<3>でなにも考慮する必要がない。
size_t ggml_number_effective_dims(const int64_t ne[GGML_MAX_DIMS]){
    size_t dims = GGML_MAX_DIMS;
    while(dims > 0 && ne[dims - 1] == 1){
        dims --;
    }
    return dims;
}

// worldからglobalとlocalを調整する。
// ->フェーズ1 一番大きい次元に割り当てを行う。
// 次元数に応じて割り振りを行う。
//
// hppに記述してテンプレートも試したが上手くいかない…
void adjustment_local(sycl::range<3> &local, const sycl::range<3> world,
//template <int Dimensions>
//void adjustment_local(sycl::range<Dimensions> &local, const sycl::range<Dimensions> world,
    const int group_size, const int sub_group_size){
    GGML_SYCL_DEBUG("[SYCL] %s world(%zu, %zu, %zu) group:%d sub:%d\n", __func__, world[0], world[1], world[2], group_size, sub_group_size);
    //GGML_SYCL_DEBUG("[SYCL] %s sizeof:%zu extend:%zu\n", __func__, sizeof(world), std::extent<decltype(world)>::value);
    //GGML_SYCL_DEBUG("[SYCL] %s size():%zu\n", __func__, world.size());
    // 一番数字の大きい次元を求める
    int max_idx = 0;
    for(int i = 1; i < 3; i++){
    //for(int i = 1; i < sizeof(world); i++){
    //for(int i = 1; i <  world.size(); i++){
        if(world[max_idx] < world[i])max_idx=i;
    }
    // 一番数字の大きい次元にworkgroupを与える。
    //for(int i=0; i < sizeof(local); i++)local[i]=(i==max_idx?group_size:1);
    for(int i=0; i < 3; i++)local[i]=(i==max_idx?group_size:1);
    //for(int i=0; i < local->size(); i++)local[i]=(i==max_idx?group_size:1);
    GGML_SYCL_DEBUG("[SYCL] %s max_idx:%d local(%zu, %zu, %zu)\n", __func__, max_idx, local[0], local[1], local[2]);
}
