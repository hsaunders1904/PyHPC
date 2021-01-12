import pyopencl as cl


def print_devices():
    for platform in cl.get_platforms():
        print(platform.name)
        for device in platform.get_devices():
            print(f"  {device.name}")
            dev_type = cl.device_type.to_string(device.type).lstrip('ALL |')
            print(f"    Type: {dev_type}")
            print(f"    Cores: {device.max_compute_units}")
            print(f"    Max Frequency: {device.max_clock_frequency} MHz")


def get_devices(device_type=None):
    devices = []
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if (device_type and device_type.upper()
                    in cl.device_type.to_string(device.type)):
                devices.append(device)
            if device_type is None:
                devices.append(device)

    return devices


def get_device(device_type=None, device_name=None):
    devices = get_devices(device_type)
    if device_name:
        devices = [
            dev for dev in devices if device_name.lower() in str(dev).lower()
        ]
        if not devices:
            print("Could not find devices matching: '{}'."
                  "".format(device_name))
            return
    if devices:
        return devices[0]
    else:
        print("Could not find devices")
