import argparse
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

import models.build_model as build_model

import os
from pathlib import Path


parser = argparse.ArgumentParser(description="TVM-Speed")

parser.add_argument("--model-name", default='deit_tiny_patch16_224',
                    choices=['deit_tiny_patch16_224',
                             'deit_small_patch16_224',
                             'deit_base_patch16_224'],
                    help="model fullname")
parser.add_argument("--log-path", default='result/deit.json',
                    help="log_file path (.json)")
parser.add_argument("--target", default='cuda -model=2080ti',
                    help="tvm target")


def main():
    args = parser.parse_args()

    # Set target device
    target = tvm.target.Target(args.target)

    # Path to save tuning log
    log_file = args.log_path

    # Load model
    name = args.model_name
    batch_size = 1
    image_shape = (3, 224, 224)
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    data_layout = "NCHW"
    kernel_layout = "OIHW"

    mod, params = build_model.get_workload(name=name,
                                        batch_size=batch_size,
                                        image_shape=image_shape,
                                        dtype="int8",
                                        data_layout=data_layout,
                                        kernel_layout=kernel_layout)

    ###################################################################################
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    ####################################################################################
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=500, timeout=1000)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=50000, 
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

    ####################################################################################
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('int8'))
    module.set_input("data", data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=1000, min_repeat_ms=500))


if __name__ == "__main__":
    main()