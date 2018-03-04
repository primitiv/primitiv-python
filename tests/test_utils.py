import sys


def available_devices():
    devices = []
    devices.extend(available_naive_devices())
    devices.extend(available_cuda_devices())
    devices.extend(available_eigen_devices())
    devices.extend(available_opencl_devices())
    for dev in devices:
        dev.dump_description()
    return devices


def available_naive_devices():
    from primitiv.devices import Naive
    return [
        Naive(),
        Naive(),
    ]


def available_cuda_devices():
    try:
        from primitiv.devices import CUDA
    except ImportError:
        return []
    devices = []
    num_devs = CUDA.num_devices()
    num_avail_devs = 0
    for dev_id in range(num_devs):
        if CUDA.check_support(dev_id):
            devices.append(CUDA(dev_id))
            num_avail_devs += 1
            if len(devices) == 1:
                devices.append(CUDA(dev_id))
    if num_devs - num_avail_devs:
        print("%d CUDA device(s) are not supported."
              % (num_devs - num_avail_devs), file=sys.stderr)
    return devices


def available_eigen_devices():
    try:
        from primitiv.devices import Eigen
    except ImportError:
        return []
    return [
        Eigen(),
        Eigen(),
    ]


def available_opencl_devices():
    try:
        from primitiv.devices import OpenCL
    except ImportError:
        return []
    devices = []
    num_pfs = OpenCL.num_platforms();
    num_avail_devs = 0
    num_devs = 0
    for pf_id in range(num_pfs):
        num_devs += OpenCL.num_devices(pf_id)
        for dev_id in range(num_devs):
            if OpenCL.check_support(pf_id, dev_id):
                devices.append(OpenCL(pf_id, dev_id))
                num_avail_devs += 1
                if len(devices) == 1:
                    devices.append(OpenCL(pf_id, dev_id))
    if num_devs != num_avail_devs:
        print("%d OpenCL device(s) are not supported."
              % (num_devs - num_avail_devs), file=sys.stderr)
    return devices
