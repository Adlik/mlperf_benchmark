import multiprocessing
import threading
import subprocess
import time
from datetime import datetime
import os
import argparse
import array
import logging

import numpy as np
import mlperf_loadgen as lg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TVM-BERT")

max_seq_length = 384
max_query_length = 64
doc_stride = 128
in_queue_cnt = 0
out_queue_cnt = 0

bs_step = 8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument(
        "--batching", choices=["Fixed", "Dynamic", "Adaptive"], default="Adaptive", help="Batching method")
    parser.add_argument("--batch-size", default=1, type=int, help="batch_size")
    parser.add_argument("--num-instance", default=1,
                        type=int, help="number of instance")
    parser.add_argument("--num-phy-cpus", default=24,
                        type=int, help="number of physical cpus")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model")
    parser.add_argument("--mlperf-conf", default="mlperf.conf",
                        help="mlperf rules config")
    parser.add_argument("--user-conf", default="user.conf",
                        help="user rules config")
    parser.add_argument("--perf-count", default=None, help="perf count")
    parser.add_argument("--profile", action="store_true",
                        help="whether enable profiler")
    parser.add_argument("--warmup", action="store_true",
                        help="whether do warmup")
    parser.add_argument("--perf_calibrate", action="store_true",
                        help="whether do performance calibration")
    parser.add_argument("--path_to_model", default="bert_ansor_11840_fastmath.so",
                        help="Path to an .xml file with a trained model")
    parser.add_argument(
        "--log-dir", default="/data/mlperf_log", help="log file directory")
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


batches = None


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, lock, init_counter, proc_idx, world_size, args):
        multiprocessing.Process.__init__(self)
        global num_ins
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.world_size = world_size
        self.args = args
        self.affinity = range(round(proc_idx * num_cpus / num_ins),
                              round((proc_idx + 1) * num_cpus / num_ins))
        self.start_core_idx = proc_idx * num_cpus // num_ins
        self.end_core_idx = (proc_idx + 1) * num_cpus // num_ins - 1
        self.length_list = {}
        self.length_time_list = {}

    def warmup(self, model, data_set, scenario):
        if self.proc_idx == 0:
            print('Start warmup...')
        count = 0

        for start in range(0, 10):
            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            eval_features = data_set.eval_features[start]

            if scenario == 'Offline':
                # only support warmup of adaptive batching
                for i in range(self.args.batch_size):
                    input_ids_list.append(eval_features.input_ids)
                    input_mask_list.append(eval_features.input_mask)
                    segment_ids_list.append(eval_features.segment_ids)
                if self.proc_idx == 0:
                    print("warmup seqlen {} batchsize {}".format(
                        max_seq_length, self.args.batch_size))
            else:
                input_ids_list.append(eval_features.input_ids)
                input_mask_list.append(eval_features.input_mask)
                segment_ids_list.append(eval_features.segment_ids)

            input_ids = np.array(input_ids_list)
            input_mask = np.array(input_mask_list)
            segment_ids = np.array(segment_ids_list)
            # warm up primitive once
            model.module.set_input(
                input_ids=input_ids, segment_ids=segment_ids, input_mask=input_mask)
            model.module.run()
            count += 1
            if count % 10 == 0 and self.proc_idx == 0:
                print('Warmup {} samples'.format(count))
        if self.proc_idx == 0:
            print('Warmup done')

    def run(self):
        cmd = "taskset -p -c %d-%d %d" % (self.start_core_idx,
                                          self.end_core_idx, self.pid)
        print(cmd)
        os.system(cmd)

        #from numexpr.utils import set_num_threads
        # set_num_threads(28)
        os.environ['OMP_NUM_THREADS'] = '{}'.format(
            self.end_core_idx-self.start_core_idx+1)

        model = BERTModel(self.args.quantized, self.args.path_to_model)
        data_set = BERTDataSet()

        if self.args.warmup:
            self.warmup(model, data_set, self.args.scenario)

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        cur_step = 0

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
            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            for sample_index in sample_index_list:
                eval_features = data_set.eval_features[sample_index]

                input_ids_list.append(eval_features.input_ids)
                input_mask_list.append(eval_features.input_mask)
                segment_ids_list.append(eval_features.segment_ids)

            input_ids = np.array(input_ids_list)
            input_mask = np.array(input_mask_list)
            segment_ids = np.array(segment_ids_list)

            model.module.set_input(
                input_ids=input_ids, segment_ids=segment_ids, input_mask=input_mask)
            model.module.run()
            out_start_np = model.module.get_output(0).asnumpy()
            start_result = Output(query_id_list, out_start_np)
            # out_end_np = model.module.get_output(1).asnumpy()
            # end_result = Output(query_id_list, out_end_np)
            self.result_queue.put(start_result)
            # self.result_queue.put(end_result)
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
        self.max_seq_len = max_seq_length

    def put(self, query_samples):
        global in_queue_cnt

        num_samples = len(query_samples)

        def idx_len(e):
            idx = e.index
            feature = self.data_set.eval_features[idx]
            return len(feature.input_ids)

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
                batch_seq_len = self.batch_size * self.max_seq_len
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
                    base_index = next_index
                print('pid-{1}: enqueued {0} batches, pad ratio = {2}%'
                      .format(num_batches, os.getpid(), (total_len-true_total_len)*100/true_total_len))
            elif batching == 'Adaptive':
                batch_seq_len = self.batch_size * self.max_seq_len
                base_index = 0
                num_batches = 0
                while base_index < num_samples:
                    base_len = idx_len(query_samples[base_index])
                    next_index = base_index + self.batch_size
                    if next_index > num_samples:
                        next_index = num_samples
                    total_len += base_len * (next_index-base_index)
                    enqueue_batch(next_index-base_index, base_index)
                    num_batches += 1
                    base_index = next_index
                print('pid-{1}: enqueued {0} batches, pad ratio = {2}%'
                      .format(num_batches, os.getpid(), (total_len-true_total_len)*100/true_total_len))
            else:
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
        for i, o in enumerate(out_list):
            response_array = array.array(
                "B", np.array(o).astype(np.float32).tobytes())
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(
                query_id_list[i], bi[0], bi[1])]
            out_queue_cnt += 1
            lg.QuerySamplesComplete(responses)


class BERTModel():
    def __init__(self, quantized, path_to_model):

        import tvm
        from tvm.contrib import graph_executor

        log.info('Loading TVM model...')
        ctx = tvm.cpu()
        lib = tvm.runtime.load_module(path_to_model)
        self.module = graph_executor.GraphModule(lib["default"](ctx))


class BERTDataSet():
    def __init__(self, total_count_override=None, perf_count_override=None, cache_path='eval_features.pickle'):
        import pickle
        from transformers import BertTokenizer
        from create_squad_data import read_squad_examples, convert_examples_to_features

        print("Constructing QSL...")
        eval_features = []
        # Load features if cached, convert from examples otherwise.
        if os.path.exists(cache_path):
            print("Loading cached features from '%s'..." % cache_path)
            with open(cache_path, 'rb') as cache_file:
                eval_features = pickle.load(cache_file)
        else:
            print(
                "No cached features at '%s'... converting from examples..." % cache_path)

            print("Creating tokenizer...")
            tokenizer = BertTokenizer("vocab.txt")

            print("Reading examples...")
            eval_examples = read_squad_examples(input_file="squad-v1.1/dev-v1.1.json",
                                                is_training=False, version_2_with_negative=False)

            print("Converting examples to features...")

            def append_feature(feature):
                eval_features.append(feature)

            convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=False,
                output_fn=append_feature,
                verbose_logging=False)

            print("Caching features at '%s'..." % cache_path)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(eval_features, cache_file)

        self.eval_features = eval_features
        self.count = total_count_override or len(self.eval_features)
        self.perf_count = perf_count_override or self.count


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

    def task_done(self):
        return self._jq.task_done()

    def join(self):
        return self._jq.join()


def main():
    global num_ins
    global num_cpus
    global in_queue_cnt
    global out_queue_cnt
    global batching
    global bs_step

    args = get_args()
    log.info(args)
    scenario = args.scenario
    accuracy_mode = args.accuracy
    perf_count = args.perf_count
    batch_size = args.batch_size
    num_ins = args.num_instance
    num_cpus = args.num_phy_cpus
    batching = args.batching

    ## TODO, remove
    log.info('Run with {} instance on {} cpus: '.format(num_ins, num_cpus))

    # Establish communication queues
    lock = multiprocessing.Lock()
    init_counter = multiprocessing.Value("i", 0)
    out_queue = multiprocessing.Queue()
    in_queue = MultiprocessShapeBasedQueue()

    # Start consumers
    consumers = [Consumer(in_queue, out_queue, lock, init_counter, i, num_ins, args)
                 for i in range(num_ins)]
    for c in consumers:
        c.start()

    # used by constructQSL
    data_set = BERTDataSet()
    issue_queue = InQueue(in_queue, batch_size, data_set)

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
    log_path = os.path.join(args.log_dir, str(datetime.now()))
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

    if accuracy_mode:
        cmd = "python accuracy-squad.py --log_file={}/mlperf_log_accuracy.json".format(
            log_path)
        subprocess.check_call(cmd, shell=True)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == '__main__':
    main()
