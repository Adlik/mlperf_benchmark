import multiprocessing
import threading
import subprocess
import time
import datetime
import os
import sys
import argparse
import array
import logging

import numpy as np
import mlperf_loadgen as lg
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ONNX-RUNTIME-DYNAMIC-SHAPE-BERT")

num_cpus = 28
num_ins = 2
NANO_SEC = 1e9
MILLI_SEC = 1000

in_queue_cnt = 0
out_queue_cnt = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument(
        "--batching", choices=["Dynamic", "NaiveBucket"], default="Adaptive", help="Batching method")
    parser.add_argument("--batch-size", default=1, type=int, help="batch_size")
    parser.add_argument("--num-instance", default=2,
                        type=int, help="number of instance")
    parser.add_argument("--num-phy-cpus", default=28,
                        type=int, help="number of physical cpus")
    parser.add_argument("--vocab", default='converted_from_tf_to_mxnet/tf.vocab',
                        type=str, help="vocab file path")
    parser.add_argument("--params", default='converted_from_tf_to_mxnet/tf_fp32.params',
                        type=str, help="FP32 params path")
    parser.add_argument("--quantized_model_prefix",
                        default='converted_from_tf_to_mxnet/offline_model/model_bert_squad_quantized_customize',
                        type=str, help="quantized model prefix")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model")
    parser.add_argument("--mlperf-conf", default="./conf/mlperf.conf",
                        help="mlperf rules config")
    parser.add_argument("--user-conf", default="./conf/user.conf",
                        help="user rules config")
    parser.add_argument("--perf-count", default=None, help="perf count")
    parser.add_argument("--profile", action="store_true",
                        help="whether enable profiler")
    parser.add_argument("--perf_calibrate", action="store_true",
                        help="whether do performance calibration")
    parser.add_argument("--log-dir", default="build/logs",
                        help="log file directory")
    parser.add_argument(
        "--path_to_model", default="dynamic_bert_opt1.onnx", help="onnx model path")
    parser.add_argument(
        "--dataset", default="dev-v1.1.json", help="dataset path")
    args = parser.parse_args()
    return args


scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def load_query_samples(sample_list):
    # This is model specific place holder
    pass


def unload_query_samples(sample_list):
    # This is model specific place holder
    pass


def block_until(counter, num_ins, t=1):
    while counter.value < num_ins:
        time.sleep(t)


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, lock, init_counter, calibrate_counter, proc_idx, world_size, args):
        multiprocessing.Process.__init__(self)
        global num_ins
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.init_counter = init_counter
        self.calibrate_counter = calibrate_counter
        self.proc_idx = proc_idx
        self.world_size = world_size
        self.args = args
        self.affinity = range(round(proc_idx * num_cpus / num_ins),
                              round((proc_idx + 1) * num_cpus / num_ins))
        self.start_core_idx = proc_idx * num_cpus // num_ins
        self.end_core_idx = (proc_idx + 1) * num_cpus // num_ins - 1
        self.length_list = {}
        self.length_time_list = {}

    def run(self):
        global batching
        cmd = "taskset -p -c %d-%d %d" % (self.start_core_idx,
                                          self.end_core_idx, self.pid)
        print(cmd)
        os.system(cmd)

        os.environ['OMP_NUM_THREADS'] = '{}'.format(
            self.end_core_idx-self.start_core_idx+1)

        model = BERTModel(self.args.path_to_model)
        data_set = BERTDataSet(self.args.dataset, self.args.perf_count)

        self.lock.acquire()
        self.calibrate_counter.value += 1
        self.lock.release()

        block_until(self.calibrate_counter, self.world_size)

        self.lock.acquire()
        self.calibrate_counter.value += 1
        self.lock.release()

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        cur_step = 0
        start_step = 384
        end_step = -1
        while True:

            next_task = self.task_queue.get(self.proc_idx)
            if next_task is None:
                # None means shutdown
                log.info(
                    'Exiting {}-pid:{}, cur_step={}'.format(self.name, self.pid, cur_step))
                self.task_queue.task_done()
                break

            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list
            inputs_list = []
            token_types_list = []
            valid_length_list = []
            for sample_index in sample_index_list:
                eval_feature = data_set.eval_features[sample_index]
                inputs, token_types, valid_length = eval_feature
                inputs_list.append(inputs)
                token_types_list.append(token_types)
                valid_length_list.append(valid_length)

            input_ids = np.array(inputs_list, dtype=np.int64)
            input_mask = np.array(token_types_list, dtype=np.int64)
            segment_ids = np.array(valid_length_list, dtype=np.int64)

            # print('progressing batch {} length {}'.format(
            #    len(input_ids), len(inputs)))

            print(input_ids.shape, input_mask.shape, segment_ids.shape)

            out = model.net.run(None, {
                                'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids})

            out = np.concatenate((out[0].reshape(*out[0].shape, 1),
                                  out[1].reshape(*out[1].shape, 1)), axis=2)
            '''
            out = np.ndarray([len(input_ids), len(inputs), 2], dtype='float32')
            '''
            result = Output(query_id_list, out)
            self.result_queue.put(result)
            self.task_queue.task_done()


class Input(object):
    def __init__(self, id_list, index_list, sample_length_list):
        assert isinstance(id_list, list)
        assert isinstance(index_list, list)
        assert isinstance(sample_length_list, list)
        assert len(id_list) == len(index_list)
        self.query_id_list = id_list
        self.sample_index_list = index_list
        self.sample_length_list = sample_length_list


class Output(object):
    def __init__(self, query_id_list, result):
        self.query_id_list = query_id_list
        self.result = result


class InQueue():
    def __init__(self, in_queue, batch_size, data_set):
        self.in_queue = in_queue
        self.batch_size = batch_size
        self.query_id_list = []
        self.sample_index_list = []
        self.sample_length_list = []
        self.index = 0
        self.data_set = data_set

    def put(self, query_samples):
        global in_queue_cnt
        ##TODO, debug
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        query_len = len(query_samples)

        num_samples = len(query_samples)

        def idx_len(e):
            idx = e.index
            feature = self.data_set.eval_features[idx]
            inputs, _, _ = feature
            return len(inputs)

        if num_samples == 1:
            if self.batch_size == 1:
                in_queue_cnt += 1
                self.in_queue.put(Input([query_samples[0].id],
                                        [query_samples[0].index],
                                        [idx_len(query_samples[0])]))
            else:
                self.index += 1
                if self.index < self.batch_size:
                    self.query_id_list.append(query_samples[0].id)
                    self.sample_index_list.append(query_samples[0].index)
                    self.sample_length_list.append(idx_len(query_samples[0]))
                else:
                    self.query_id_list.append(query_samples[0].id)
                    self.sample_index_list.append(query_samples[0].index)
                    self.sample_length_list.append(idx_len(query_samples[0]))
                    self.in_queue.put(
                        Input(self.query_id_list, self.sample_index_list, self.sample_length_list))
                    in_queue_cnt += self.batch_size
                    self.index = 0
                    self.query_id_list = []
                    self.sample_index_list = []
                    self.sample_length_list = []
        else:

            query_samples.sort(key=idx_len, reverse=True)

            def enqueue_batch(cur_batch_size, base_index=0):
                global in_queue_cnt
                id_list = []
                index_list = []
                length_list = []
                for i in range(cur_batch_size):
                    id_list.append(query_samples[base_index + i].id)
                    index_list.append(query_samples[base_index + i].index)
                    length_list.append(idx_len(query_samples[base_index + i]))
                self.in_queue.put(Input(id_list, index_list, length_list))
                in_queue_cnt += cur_batch_size

            global batching
            true_total_len = 0
            total_len = 0
            for i in range(num_samples):
                true_total_len += idx_len(query_samples[i])
            if batching == 'Dynamic':
                batch_seq_len = None
                base_index = 0
                num_batches = 0
                while base_index < num_samples:
                    base_len = idx_len(query_samples[base_index])
                    for i in range(base_index, num_samples):
                        current_len = base_len * (i-base_index+1)
                        if i+1 < num_samples:
                            next_len = base_len * (i+1-base_index+1)
                            if next_len > batch_seq_len:
                                if next_len - batch_seq_len > batch_seq_len - current_len:
                                    next_index = i+1
                                else:
                                    next_index = i+2
                                break
                        else:
                            next_index = i+1
                            break
                    total_len += base_len * (next_index-base_index)
                    enqueue_batch(next_index-base_index, base_index)
                    num_batches += 1
                    # print('pid-{2}: enqueue bs={0} and input volume {1}...'
                    #    .format(next_index-base_index, current_len, os.getpid()))
                    base_index = next_index
                print('pid-{1}: enqueued {0} batches, pad ratio = {2}%'
                      .format(num_batches, os.getpid(), (total_len-true_total_len)*100/true_total_len))
            elif batching == 'NaiveBucket':
                bucket_stats = {}
                for q in query_samples:
                    leng = idx_len(q)
                    if leng not in bucket_stats:
                        bucket_stats[leng] = [[q.id], [q.index]]
                    else:
                        bucket_stats[leng][0].append(q.id)
                        bucket_stats[leng][1].append(q.index)

                for leng in sorted(bucket_stats.keys(), reverse=True):
                    id_list, index_list = bucket_stats[leng]
                    batch_size = len(id_list)
                    length_list = [leng] * len(id_list)
                    if batch_size > 3000:
                        step_batch = 256
                        for i in range(0, batch_size, step_batch):
                            id_list_step = id_list[i: i+step_batch]
                            index_list_step = index_list[i: i+step_batch]
                            length_list_step = [leng] * len(id_list_step)
                            self.in_queue.put(
                                Input(id_list_step, index_list_step, length_list_step))
                            in_queue_cnt += len(id_list_step)
                    else:
                        self.in_queue.put(
                            Input(id_list, index_list, length_list))
                        in_queue_cnt += len(id_list)
                # print(in_queue_cnt)

            elif batching == 'Fixed':
                num_batch = num_samples // self.batch_size
                remaining_batch = num_samples % self.batch_size
                ## TODO, remove
                print('pid-{3}: split the datasets into {0} batches with bs={1} and remaining {2}...'
                      .format(num_batch, self.batch_size, remaining_batch, os.getpid()))

                for b in range(num_batch):
                    base_index = b * self.batch_size
                    enqueue_batch(self.batch_size, base_index)

                if remaining_batch > 0:
                    base_index = num_batch * self.batch_size
                    enqueue_batch(remaining_batch, base_index)

            else:
                raise('Unknown batching method {}'.format(batching))

        #print ('in_queue_cnt=', in_queue_cnt)


def flush_queries():
    pass


def process_latencies(latencies_ns):
    # It's called by loadgen to show us the recorded latencies
    log.info("Average latency (ms) per query:")
    log.info(np.mean(latencies_ns)/1000000.0)
    log.info("Median latency (ms): ")
    log.info(np.percentile(latencies_ns, 50)/1000000.0)
    log.info("90 percentile latency (ms): ")
    log.info(np.percentile(latencies_ns, 90)/1000000.0)


def response_loadgen(out_queue):
    global out_queue_cnt
    while True:
        next_task = out_queue.get()
        if next_task is None:
            # None means shutdown
            log.info('Exiting response thread')
            break
        query_id_list = next_task.query_id_list
        result = next_task.result

        batch_size = len(query_id_list)
        result.reshape(batch_size, -1, 2)

        out_list = np.split(result, batch_size, axis=0)
        #responses = []
        for i, o in enumerate(out_list):
            response_array = array.array(
                "B", np.array(o).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(
                query_id_list[i], bi[0], bi[1])]
            out_queue_cnt += 1
            #print('Response loadgen ({}), query_id {}, out_queue_cnt {}'.format(os.getpid(), query_id_list[i], out_queue_cnt))
            lg.QuerySamplesComplete(responses)
        print('progressing over: {}/{}, batch {} length {}'.format(out_queue_cnt,
              in_queue_cnt, batch_size, result.shape[1]))

        # lg.QuerySamplesComplete(responses)


class BERTModel:
    def __init__(self, path_to_model):
        self.net = ort.InferenceSession(path_to_model, providers=[
                                        'DnnlExecutionProvider', 'CPUExecutionProvider'])


class BERTDataSet:
    def __init__(self, data_path, perf_count):
        import torch
        dataset = torch.load(data_path)
        feature_names = ['input_ids_samples',
                         'input_mask_samples', 'segment_ids_samples']

        self.eval_features = [[dataset[fn][i].numpy() for fn in feature_names]
                              for i in range(len(dataset[feature_names[0]]))]
        self.count = len(self.eval_features)
        self.perf_count = perf_count if perf_count is not None else self.count


class MultiprocessShapeBasedQueue(object):
    def __init__(self):
        global num_ins
        self._jq = multiprocessing.JoinableQueue()
        self._instances_queue = [multiprocessing.Queue()
                                 for _ in range(num_ins)]
        self._manager = multiprocessing.Manager()
        self.shape_in_instance = self._manager.dict()
        self.finish_status = self._manager.dict()

    def get(self, instance_id):
        return self._jq.get()

    def put(self, obj, block=True, timeout=None):
        return self._jq.put(obj, block, timeout)
        ##print("end put")

    def task_done(self):
        # print("task_done")
        return self._jq.task_done()
        #print("end task_done")

    def join(self):
        # print("join")
        return self._jq.join()
        #print("end join")


def main():
    global num_ins
    global num_cpus
    global in_queue_cnt
    global out_queue_cnt
    global batching

    args = get_args()
    log.info(args)
    scenario = args.scenario
    accuracy_mode = args.accuracy
    perf_count = args.perf_count
    batch_size = args.batch_size
    num_ins = args.num_instance
    num_cpus = args.num_phy_cpus
    batching = args.batching

    log.info('Run with {} instance on {} cpus: '.format(num_ins, num_cpus))

    # Establish communication queues
    lock = multiprocessing.Lock()
    init_counter = multiprocessing.Value("i", 0)
    calibrate_counter = multiprocessing.Value("i", 0)
    out_queue = multiprocessing.Queue()
    in_queue = MultiprocessShapeBasedQueue()

    # Start consumers
    consumers = [Consumer(in_queue, out_queue, lock, init_counter, calibrate_counter, i, num_ins, args)
                 for i in range(num_ins)]
    for c in consumers:
        c.start()

    # used by constructQSL
    data_set = BERTDataSet(args.dataset, args.perf_count)
    issue_queue = InQueue(in_queue, batch_size, data_set)

    # Wait until all sub-processors ready to do calibration
    block_until(calibrate_counter, num_ins)
    # Wait until all sub-processors done calibration
    block_until(calibrate_counter, 2*num_ins)
    # Wait until all sub-processors are ready
    block_until(init_counter, num_ins)

    # Start response thread
    response_worker = threading.Thread(
        target=response_loadgen, args=(out_queue,))
    response_worker.daemon = True
    response_worker.start()

    # Start loadgen
    settings = lg.TestSettings()
    settings.scenario = scenario_map[scenario]
    settings.FromConfig(args.mlperf_conf, "bert", scenario)
    settings.FromConfig(args.user_conf, "bert", scenario)
    settings.mode = lg.TestMode.AccuracyOnly if accuracy_mode else lg.TestMode.PerformanceOnly

    def issue_queries(query_samples):
        # It's called by loadgen to send query to SUT
        issue_queue.put(query_samples)

    sut = lg.ConstructSUT(
        issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(
        data_set.count, data_set.perf_count, load_query_samples, unload_query_samples)

    log_path = os.path.join(args.log_dir, str(datetime.datetime.now()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    #lg.StartTest(sut, qsl, settings)
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)

    # Wait until outQueue done
    while out_queue_cnt < in_queue_cnt:
        time.sleep(0.2)

    in_queue.join()

    for i in range(num_ins):
        in_queue.put(None)

    for c in consumers:
        c.join()

    out_queue.put(None)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == '__main__':
    main()
