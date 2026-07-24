from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from zlc_neutral_atom.devices.camera._dcam_driver import DcamProperty, DcamValue
from zlc_neutral_atom.devices.camera.dcam import (
    DcamCameraAdapter,
    DcamCameraConfig,
    DcamCaptureInterrupted,
)


class _FakeDcamDevice:
    def __init__(self, calls: list[tuple[str, int]]) -> None:
        self.calls = calls
        self.sensor_shape = (12, 16)
        self.properties = {
            DcamProperty.EXPOSURE_TIME: 0.02,
            DcamProperty.TRIGGER_SOURCE: int(DcamValue.TRIGGER_SOURCE_EXTERNAL),
            DcamProperty.TRIGGER_ACTIVE: int(DcamValue.TRIGGER_ACTIVE_EDGE),
            DcamProperty.TRIGGER_POLARITY: int(DcamValue.TRIGGER_POLARITY_POSITIVE),
            DcamProperty.TRIGGER_GLOBAL_EXPOSURE: 1,
            DcamProperty.READOUT_SPEED: 1,
            DcamProperty.SENSOR_MODE: 1,
            DcamProperty.BINNING: 1,
            DcamProperty.SUBARRAY_HPOS: 0,
            DcamProperty.SUBARRAY_HSIZE: 16,
            DcamProperty.SUBARRAY_VPOS: 0,
            DcamProperty.SUBARRAY_VSIZE: 12,
            DcamProperty.SUBARRAY_MODE: int(DcamValue.MODE_OFF),
            DcamProperty.TIMING_MIN_TRIGGER_INTERVAL: 0.025,
            DcamProperty.TIMING_GLOBAL_EXPOSURE_DELAY: 0.001,
            DcamProperty.IMAGE_PIXEL_TYPE: int(DcamValue.PIXEL_MONO16),
        }
        self.ring_size = 0
        self.ring: dict[int, tuple[np.ndarray, int, int, int, int]] = {}
        self.count = 0
        self.newest = -1
        self.started = False
        self.released = 0
        self.stop_error: BaseException | None = None
        self.transfer_error: BaseException | None = None
        self.advance_after_copy: tuple[int, int] | None = None
        self._condition = threading.Condition()

    def _sdk(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def _frame_geometry(self) -> tuple[int, int]:
        binning = int(self.properties[DcamProperty.BINNING])
        if self.properties[DcamProperty.SUBARRAY_MODE] == int(DcamValue.MODE_ON):
            width = int(self.properties[DcamProperty.SUBARRAY_HSIZE])
            height = int(self.properties[DcamProperty.SUBARRAY_VSIZE])
        else:
            height, width = self.sensor_shape
        return height // binning, width // binning

    def get_property(self, property_id: DcamProperty) -> float:
        self._sdk(f"get:{property_id.name}")
        if property_id is DcamProperty.IMAGE_WIDTH:
            return float(self._frame_geometry()[1])
        if property_id is DcamProperty.IMAGE_HEIGHT:
            return float(self._frame_geometry()[0])
        return float(self.properties[property_id])

    def set_get_property(self, property_id: DcamProperty, value: float) -> float:
        self._sdk(f"set:{property_id.name}")
        self.properties[property_id] = float(value)
        return float(value)

    def property_attributes(self, property_id: DcamProperty) -> tuple[float, float]:
        self._sdk(f"attr:{property_id.name}")
        if property_id is DcamProperty.SUBARRAY_HSIZE:
            return 4.0, float(self.sensor_shape[1])
        if property_id is DcamProperty.SUBARRAY_VSIZE:
            return 4.0, float(self.sensor_shape[0])
        raise AssertionError(property_id)

    def allocate_buffer(self, frame_count: int) -> None:
        self._sdk("allocate")
        self.ring_size = int(frame_count)
        self.ring.clear()

    def release_buffer(self) -> None:
        self._sdk("release")
        self.released += 1
        self.ring_size = 0
        self.ring.clear()

    def start_capture(self) -> None:
        self._sdk("start")
        self.started = True
        self.count = 0
        self.newest = -1

    def stop_capture(self) -> None:
        self._sdk("stop")
        if self.stop_error is not None:
            raise self.stop_error
        self.started = False

    def capture_status(self) -> int:
        self._sdk("status")
        return 1 if self.started else 2

    def transfer_info(self) -> tuple[int, int]:
        self._sdk("transfer")
        if self.transfer_error is not None:
            raise self.transfer_error
        with self._condition:
            return self.count, self.newest

    def copy_frame(self, ring_index: int):
        self._sdk(f"copy:{ring_index}")
        with self._condition:
            result = self.ring[ring_index]
            if self.advance_after_copy is not None:
                self.count, self.newest = self.advance_after_copy
                self.advance_after_copy = None
            return result

    def wait_frame_ready(self, timeout_milliseconds: int) -> bool:
        self._sdk("wait")
        with self._condition:
            before = self.count
            self._condition.wait(timeout_milliseconds / 1000.0)
            return self.count > before

    def close(self) -> None:
        self._sdk("close")

    def publish(self, values: tuple[int, ...], *, newest: int) -> None:
        """Test producer: install one complete DCAM transfer snapshot."""

        with self._condition:
            count = len(values)
            assert self.ring_size > 0 and 0 <= newest < self.ring_size
            for ordinal, value in enumerate(values):
                distance = count - 1 - ordinal
                index = (newest - distance) % self.ring_size
                self.ring[index] = (
                    np.full(self._frame_geometry(), value, dtype=np.uint16),
                    100 + ordinal,
                    200 + ordinal,
                    1_000 + ordinal,
                    2_000 + ordinal,
                )
            self.count = count
            self.newest = newest
            self._condition.notify_all()


class _FakeDcamDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.device = _FakeDcamDevice(self.calls)

    def initialize(self) -> bool:
        self.calls.append(("initialize", threading.get_ident()))
        return True

    def uninitialize(self) -> None:
        self.calls.append(("uninitialize", threading.get_ident()))

    def open_device(self, index: int) -> _FakeDcamDevice:
        self.calls.append((f"open:{index}", threading.get_ident()))
        return self.device


def _config(*, roi=(4, 4, 8, 8)) -> DcamCameraConfig:
    return DcamCameraConfig(
        capture_trigger_channels=("qcmos_trigger",),
        exposure_seconds=0.01,
        roi_xywh=roi,
    )


def test_working_point_is_live_readback_and_all_sdk_calls_share_owner() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        point = adapter.capture_working_point()
        assert point.frame_shape_yx == (8, 8)
        assert point.sensor_shape_yx == (12, 16)
        assert point.roi_origin_yx == (4, 4)
        assert point.roi_shape_yx == (8, 8)
        assert point.capture_trigger_channels == ("qcmos_trigger",)
        adapter.configure_exposure_seconds(0.015)
        assert adapter.capture_working_point().exposure_seconds == 0.015
    finally:
        adapter.close()
    owner_threads = {thread_id for _name, thread_id in driver.calls}
    assert len(owner_threads) == 1
    assert next(iter(owner_threads)) != threading.get_ident()


def test_count_first_drain_uses_snapshot_newest_and_preserves_record_metadata() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        adapter.arm(
            3,
            source_group_sizes=(3,),
            buffer_frame_count=3,
            timeout=1.0,
        )
        assert driver.device.ring_size == 3
        driver.device.publish((11, 22, 33), newest=1)
        records = [
            adapter.read_frame_records(1, timeout=1.0, exact=True)[0]
            for _index in range(3)
        ]
        driver.device.ring.clear()
        assert [int(record.image[0, 0]) for record in records] == [11, 22, 33]
        assert [record.source_ordinal for record in records] == [0, 1, 2]
        assert [record.produced_count for record in records] == [1, 2, 3]
        assert [record.driver_buffer_index for record in records] == [2, 0, 1]
        assert [record.frame_stamp for record in records] == [100, 101, 102]
        assert [record.camera_stamp for record in records] == [200, 201, 202]
        assert not records[0].image.flags.writeable
        terminal = adapter.finish_record_capture()
        assert terminal.produced_count == 3
        assert adapter.finish_record_capture() is terminal
    finally:
        adapter.close()


def test_finite_arm_requires_the_complete_physical_buffer_cardinality() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        with pytest.raises(ValueError, match="must equal the complete frame count"):
            adapter.arm(
                3,
                source_group_sizes=(3,),
                buffer_frame_count=2,
                timeout=1.0,
            )
        assert driver.device.ring_size == 0
    finally:
        adapter.close()


@pytest.mark.parametrize("exact", (False, True))
def test_every_copy_rechecks_ring_overwrite(exact: bool) -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        adapter.arm(
            None,
            source_group_sizes=None,
            buffer_frame_count=3,
            timeout=1.0,
        )
        driver.device.publish((1, 2, 3), newest=1)
        driver.device.advance_after_copy = (4, 2)
        with pytest.raises(RuntimeError, match="overwrite a frame during copy"):
            adapter.read_frame_records(1, timeout=1.0, exact=exact)
    finally:
        adapter.finish_record_capture()
        adapter.close()


def test_transfer_count_rollback_fails_loudly() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        adapter.arm(
            3,
            source_group_sizes=(3,),
            buffer_frame_count=3,
            timeout=1.0,
        )
        driver.device.publish((1, 2), newest=1)
        adapter.read_frame_records(2, timeout=1.0, exact=True)
        driver.device.count = 1
        driver.device.newest = 0
        with pytest.raises(RuntimeError, match="moved backwards"):
            adapter.read_frame_records(1, timeout=0.0, exact=False)
    finally:
        driver.device.count = 2
        driver.device.newest = 1
        adapter.finish_record_capture()
        adapter.close()


def test_transfer_query_failure_never_falls_back_to_latest_frame() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        adapter.arm(
            None,
            source_group_sizes=None,
            buffer_frame_count=2,
            timeout=1.0,
        )
        driver.device.transfer_error = RuntimeError("injected transfer failure")
        with pytest.raises(RuntimeError, match="injected transfer failure"):
            adapter.read_frame_records(1, timeout=0.0, exact=False)
    finally:
        driver.device.transfer_error = None
        adapter.finish_record_capture()
        adapter.close()


def test_concurrent_finish_unblocks_read_and_freezes_one_terminal_record() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    adapter.arm(
        None,
        source_group_sizes=None,
        buffer_frame_count=2,
        timeout=1.0,
    )
    outcome: list[BaseException] = []

    def read() -> None:
        try:
            adapter.read_frame_records(1, timeout=5.0, exact=False)
        except BaseException as error:
            outcome.append(error)

    reader = threading.Thread(target=read)
    reader.start()
    deadline = time.monotonic() + 1.0
    while not any(name == "wait" for name, _thread in driver.calls):
        if time.monotonic() >= deadline:
            raise AssertionError("reader did not enter the fake DCAM wait")
        time.sleep(0.005)
    terminal = adapter.finish_record_capture()
    reader.join(1.0)
    try:
        assert not reader.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], DcamCaptureInterrupted)
        assert adapter.finish_record_capture() is terminal
        assert terminal.produced_count == 0
    finally:
        adapter.close()


def test_stop_failure_retains_driver_ring_for_explicit_recovery() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    adapter.arm(
        None,
        source_group_sizes=None,
        buffer_frame_count=2,
        timeout=1.0,
    )
    driver.device.stop_error = RuntimeError("injected stop failure")
    with pytest.raises(RuntimeError, match="injected stop failure"):
        adapter.finish_record_capture()
    assert adapter.capture_state()[0]
    assert driver.device.released == 0
    driver.device.stop_error = None
    adapter.finish_record_capture()
    adapter.close()


def test_released_ring_with_invalid_final_state_never_fabricates_terminal() -> None:
    driver = _FakeDcamDriver()
    adapter = DcamCameraAdapter(_config(), driver=driver)
    try:
        adapter.arm(
            None,
            source_group_sizes=None,
            buffer_frame_count=2,
            timeout=1.0,
        )
        driver.device.transfer_error = RuntimeError("injected final transfer failure")
        with pytest.raises(RuntimeError, match="not authoritative") as first:
            adapter.finish_record_capture()
        assert driver.device.released == 1
        calls_after_failure = tuple(driver.calls)

        driver.device.transfer_error = None
        with pytest.raises(RuntimeError, match="not authoritative") as second:
            adapter.finish_record_capture()
        assert second.value is first.value
        assert tuple(driver.calls) == calls_after_failure

        adapter.arm(
            None,
            source_group_sizes=None,
            buffer_frame_count=2,
            timeout=1.0,
        )
        assert adapter.finish_record_capture().produced_count == 0
    finally:
        adapter.close()
