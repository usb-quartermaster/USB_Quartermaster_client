#!/usr/bin/env python3
import json
import logging
import platform
import signal
import socket
import ssl
import sys
import time
from argparse import Namespace, ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from select import select
from threading import Event
from typing import Dict, List, NamedTuple, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen, Request

from USB_Quartermaster_common import plugins, AbstractLocalDriver

TEARDOWN_COMMAND = b"teardown"
TEARDOWN_ACK = b"Teardown started"
REFRESH_RETRY_LIMIT = 3
REFRESH_RETRY_SLEEP = 10

VERSION = '1.0'

logger = logging.getLogger()

error_counter = 0

START_TIMESTAMP = time.time()


def formatted_print(message: str):
    duration = time.time() - START_TIMESTAMP
    print(f"{duration:.1f} {message}")


@lru_cache()
def get_loaded_drivers() -> Dict[str, AbstractLocalDriver]:
    return {cls.IDENTIFIER: cls for cls in plugins.local_driver_classes()}


#
#  ___  _____   _____ ___ ___ ___
# |   \| __\ \ / /_ _/ __| __/ __|
# | |) | _| \ V / | | (__| _|\__ \
# |___/|___| \_/ |___\___|___|___/
#
class Device(object):
    connect_complete = False

    def __init__(self, conf: Dict):
        self.conf = conf
        self.name = conf['name']
        loaded_drivers = get_loaded_drivers()
        if conf['driver'] in loaded_drivers:
            self.driver = loaded_drivers[conf['driver']](conf=conf)
        else:
            raise SystemError(f"No driver found to support '{conf['driver']}'")

    def connect(self):
        if not self.connected():
            formatted_print(f"Connecting {self.name}")
            self.driver.connect()
            self.connect_complete = True
            formatted_print(f"Done connecting {self.name}")

    def disconnect(self):
        if self.connected():
            formatted_print(f"Disconnecting {self.name}")
            self.driver.disconnect()
            formatted_print(f"Done disconnecting {self.name}")

    def connected(self):
        return self.driver.connected()

    def __str__(self):
        return self.name


class Reservation(NamedTuple):
    devices: List[Device]
    use_password: str
    resource_url: str
    reservation_url: str
    auth_token: str = None


def manage_devices(devices: List[Device], polling_interval: int, teardown: Event):
    setup_done = False
    while not teardown.is_set():
        for device in devices:
            device.connect()  # connect() is lazy, if device is connected it won't do anything
        if not setup_done:
            formatted_print("Setup complete, keep this terminal open to keep your reservation active")
            setup_done = True
        teardown.wait(polling_interval)


def disconnect_devices(devices: List[Device]):
    """
    Disconnect devices

    """
    global error_counter
    for device in devices:
        if device.connect_complete:
            formatted_print(f"Disconnecting {device.name}")
            try:
                device.disconnect()
            except Exception as e:  # Swallowing exception so disconnection failures don't impact each other
                error_counter += 1
                formatted_print(f"Got the following exception when trying to disconnect {device.name}: {e}")
        else:
            formatted_print(f"Skipping disconnecting {device.name} as it never completed connecting")


def get_resource_status(url: str, config: Namespace, teardown: Event):
    while not teardown.is_set():
        logger.debug('+')
        refresh_successful = None
        for _ in range(0, REFRESH_RETRY_LIMIT):
            try:
                refresh_successful = refresh_reservation(url, config.auth_token, config.disable_validation)
                break  # The retry loop
            except Exception:
                teardown.wait(REFRESH_RETRY_SLEEP)
        else:
            logger.error(f"Failed to reach Quartermaster server after {REFRESH_RETRY_LIMIT} tries. Triggering teardown")
            teardown.set()

        if refresh_successful:
            teardown.wait(config.reservation_polling)
            continue
        else:
            formatted_print("Reservation expired, triggering teardown")
            teardown.set()


def preflight_checks(reservation: Reservation):
    """
    This calls every driver being used by the current reservation to perform basic checks to catch problem before we
    start attaching devices. Success is running this function not raising any Exceptions.
    """
    drivers_checked = set()
    loaded_drivers = get_loaded_drivers()
    for device in reservation.devices:
        driver_name = device.conf['driver']
        if driver_name in drivers_checked:
            continue
        if driver_name not in loaded_drivers:
            formatted_print(f"No driver found to support '{driver_name}', perhaps you mare missing a plugin. You"
                            f"might want to try 'pip install USB_Quartermaster_{driver_name}' to see if there is a"
                            f" driver")
            raise ModuleNotFoundError(f"No driver found to support {driver_name}")
        formatted_print(f"Preflight checking {driver_name}")
        device.driver.preflight_check()
        drivers_checked.add(driver_name)


#
#  ___ ___ _____   _____ ___    ___ ___  __  __ __  __ ___
# / __| __| _ \ \ / / __| _ \  / __/ _ \|  \/  |  \/  / __|
# \__ \ _||   /\ V /| _||   / | (_| (_) | |\/| | |\/| \__ \
# |___/___|_|_\ \_/ |___|_|_\  \___\___/|_|  |_|_|  |_|___/
#
class QuartermasterServerError(Exception):
    pass


def quartermaster_request(url: str, method: str,
                          token: Optional[str] = None,
                          data: Optional[bytes] = None,
                          disable_validation=False) -> [int, bytes, str]:
    headers = {'Accept': 'application/json',
               "Quartermaster_client_version": VERSION}
    if token:
        headers["Authorization"] = f'Token {token}'
    logger.debug(f"headers={headers}")

    request_args = {'url': url,
                    'method': method,
                    'headers': headers
                    }
    if data:
        request_args['data'] = data

    req = Request(**request_args)

    extra_urlopen_args = {}
    if disable_validation:
        extra_urlopen_args['context'] = ssl._create_unverified_context()
    try:
        # TODO: Probably should add a retry loop here
        response = urlopen(req, timeout=10, **extra_urlopen_args)
        http_code = response.code
        content = response.read()
        final_url = response.geturl()
        logger.debug(f"Final URL = {final_url}")
    except HTTPError as e:
        http_code = e.code
        content = e.msg,
        final_url = url
    except URLError as e:
        raise QuartermasterServerError(f"Error trying to reach quartermaster server: {e}")

    logger.debug(f"Response {http_code} {content}")

    return http_code, content, final_url


def get_quartermaster_reservation(url: str, message: Optional[str],
                                  auth_token: Optional[str] = None,
                                  disable_validation=False) -> Reservation:
    # POST method because data is being passed. Server will create a reservation, or if the token own already
    # has one it will return the already active reservation

    values = {}
    if message is not None:
        values['used_for'] = message

    data = urlencode(values).encode('utf-8')
    http_code, content, final_url = quartermaster_request(url=url,
                                                          token=auth_token,
                                                          method='POST',
                                                          data=data,
                                                          disable_validation=disable_validation)

    if http_code == 404:
        raise QuartermasterServerError(f"That reservation was not found")
    elif http_code not in [200, 201]:
        raise QuartermasterServerError(f"Got unexpected response from server when retrieving reservation. "
                                       f"HTTP STATUS={http_code}, BODY={content}")
    decoded = json.loads(content, encoding='utf-8')
    return Reservation(devices=[Device(device) for device in decoded['devices']],
                       use_password=decoded['use_password'],
                       resource_url=final_url,
                       reservation_url=url, auth_token=auth_token)


def refresh_reservation(url: str, auth_token: Optional[str] = None, disable_validation=False) -> bool:
    http_code, content, _ = quartermaster_request(url=url,
                                                  token=auth_token,
                                                  method='PATCH',
                                                  disable_validation=disable_validation)
    if http_code == 404:
        return False
    elif http_code != 202:
        raise ConnectionError(f"Unexpected response from server, HTTP CODE={http_code}, CONTENT={content}")
    return True


def cancel_reservation(url: str, auth_token: Optional[str] = None, disable_validation=False) -> bool:
    formatted_print(f"Canceling reservation for resource {url}, please wait")

    http_code, content, _ = quartermaster_request(url=url,
                                                  token=auth_token,
                                                  method='DELETE',
                                                  disable_validation=disable_validation)
    if http_code == 204:
        formatted_print("Reservation canceled successfully")
        return True
    raise ConnectionError(f"Unexpected response when canceling reservation, HTTP CODE={http_code}, CONTENT={content}")


def wait_for_commands(config: Namespace, teardown: Event):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((config.listen_ip, config.listen_port))
            logger.debug(f"Listening on {config.listen_ip}:{config.listen_port} for commands")
            sock.listen(1)
            while not teardown.is_set():
                # Check once a second and see if have any incoming connections
                readable, _, _ = select([sock], [], [], 1)
                if len(readable) == 0:
                    continue

                conn, addr = sock.accept()

                data = conn.recv(1024)
                logger.debug(f"Command received, {data}")
                if not data:
                    continue
                if data.startswith(TEARDOWN_COMMAND + b"\r") \
                        or data.startswith(TEARDOWN_COMMAND + b"\n"):
                    conn.sendall(TEARDOWN_ACK)
                    formatted_print(TEARDOWN_ACK.decode('utf-8'))
                    teardown.set()
    except Exception as e:
        formatted_print(f"Exception when trying to to start command listener: {repr(e)}")
        teardown.set()


def initiate_teardown(config: Namespace):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((config.listen_ip, config.listen_port))
        s.sendall(b'teardown\r')
        data = s.recv(100)
    formatted_print(data.decode('utf-8'))
    if data == TEARDOWN_ACK:
        exit(0)
    else:
        formatted_print(f"Unexpected response from client at {config.listen_ip}:{config.listen_port}")
        exit(1)


def start_threads(reservation: Reservation, config: Namespace):
    teardown_event = Event()

    def signal_handler(signal_num, stack_frame):
        logger.debug(f"Signal {signal_num} caught, triggering teardown")
        teardown_event.set()
        logger.debug("teardown_event.set()")

    signal.signal(signal.SIGINT, signal_handler)  # ctrl-c
    signal.signal(signal.SIGTERM, signal_handler)  # kill
    if platform.system().lower() != "windows":
        signal.signal(signal.SIGQUIT, signal_handler)  # quit
        signal.signal(signal.SIGHUP, signal_handler)  # Hangup (closed terminal?)

    with ThreadPoolExecutor() as pool:
        pool.submit(manage_devices, devices=reservation.devices, polling_interval=config.device_polling,
                    teardown=teardown_event)
        pool.submit(get_resource_status, url=reservation.resource_url, config=config, teardown=teardown_event)
        pool.submit(wait_for_commands, config=config, teardown=teardown_event)

    while not teardown_event.is_set():
        teardown_event.wait()

    disconnect_devices(reservation.devices)


def load_arguments(args: List[str], reservation_message=None, auth_token=None, reservation_url=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--listen_ip", type=str, default='127.0.0.1', help="Where to listen for local commands")
    parser.add_argument("--listen_port", type=int, default=4242, help="What port to listen for local commands")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debugging output")

    parser.add_argument("--auth_token", type=str, default=auth_token,
                        help="Quartermaster server authentication token, only needed when --reservation_url doesn't "
                             "include use credential")
    parser.add_argument("--reservation_message", type=str, help="Message displayed with reservation",
                        default=reservation_message)
    parser.add_argument("--device_polling", type=int, default=5,
                        help="How many seconds to wait between checks to ensure devices are connected")
    parser.add_argument("--reservation_polling", type=int, default=60,
                        help="How many seconds to wait between checks to ensure resource reservation is still active")
    parser.add_argument("--disable_validation", action='store_true',
                        help="Disable TLS validation of server certificates")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stop_client", action='store_true',
                       help="Stop the Quartermaster client. uses the --listen-* arguments if present")
    group.add_argument('quartermaster_url', metavar='quartermaster_url', type=str, default=reservation_url,
                       help='URL to quartermaster server reservation', nargs='?')
    all_parsed = parser.parse_args(args)
    return all_parsed


def main(args: List[str]):
    config = load_arguments(args)

    # Setup logging
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)

    if config.stop_client:
        initiate_teardown(config=config)
        return

    reservation = None
    try:
        reservation = get_quartermaster_reservation(url=config.quartermaster_url,
                                                    auth_token=config.auth_token,
                                                    message=config.reservation_message,
                                                    disable_validation=config.disable_validation)
        formatted_print(f"Reservation active for resource {reservation.resource_url}")
        preflight_checks(reservation)

        start_threads(reservation, config)

        formatted_print("Cleanup done")
    except Exception as e:
        formatted_print(str(e))
        if reservation is not None:
            try:
                cancel_reservation(url=reservation.reservation_url, auth_token=reservation.auth_token,
                                   disable_validation=config.disable_validation)
            except Exception as ee:
                formatted_print(f"We got an exception while trying to cancel our reservation: {ee}")

        if config.debug:  # So we get a stack traces
            raise e
        exit(1)

    if reservation is not None:
        cancel_reservation(url=reservation.reservation_url, auth_token=reservation.auth_token,
                           disable_validation=config.disable_validation)

    exit(error_counter)


if __name__ == '__main__':
    main(sys.argv[1:])
