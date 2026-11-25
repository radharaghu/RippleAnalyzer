import numpy as np
import pytest

from ripple_analyzer.core.batch_processor import BatchProcessor, write_results_csv
from ripple_analyzer.core.file_handlers import FileLoader
from ripple_analyzer.demo import DEMO_RECORDINGS, write_demo_dataset


def test_csv_with_time_column_infers_sampling_rate(tmp_path):
    p = tmp_path / "rec.csv"
    t = np.arange(1000) / 1250.0
    np.savetxt(p, np.column_stack([t, np.sin(t)]), delimiter=",", header="time_s,lfp", comments="")
    signal, fs = FileLoader.load(p)
    assert fs == pytest.approx(1250.0)
    assert len(signal) == 1000


def test_headerless_csv_keeps_first_sample(tmp_path):
    p = tmp_path / "raw.csv"
    np.savetxt(p, np.array([-21.5, 3.0, 4.0]))
    signal, fs = FileLoader.load(p, fs=2000)
    assert signal.tolist() == [-21.5, 3.0, 4.0]
    assert fs == 2000


def test_non_numeric_csv_gives_clear_error(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("File,Ripples\n/a/b.csv,8\n")
    with pytest.raises(ValueError, match="non-numeric"):
        FileLoader.load(p)


def test_multichannel_binary_selects_one_channel(tmp_path):
    p = tmp_path / "rec.bin"
    data = np.arange(12, dtype=np.int16).reshape(6, 2)  # (time, channels)
    data.tofile(p)
    signal, _ = FileLoader.load(p, n_channels=2, channel=1)
    assert signal.tolist() == data[:, 1].tolist()


def test_nwb_electrical_series_roundtrip(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from pynwb.ecephys import ElectricalSeries

    nwb = pynwb.NWBFile("test", "id", datetime.now(timezone.utc))
    dev = nwb.create_device("probe")
    grp = nwb.create_electrode_group("shank", "", "CA1", dev)
    for _ in range(2):
        nwb.add_electrode(group=grp, location="CA1")
    region = nwb.create_electrode_table_region([0, 1], "both")
    data = np.random.default_rng(0).standard_normal((5000, 2))
    nwb.add_acquisition(ElectricalSeries(name="lfp", data=data, electrodes=region, rate=1250.0))
    path = tmp_path / "rec.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)

    signal, fs = FileLoader.load(path, channel=1)
    assert fs == 1250.0
    np.testing.assert_allclose(signal, data[:, 1])


def test_unsupported_and_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileLoader.load(tmp_path / "nope.csv")
    (tmp_path / "x.txt").write_text("1")
    with pytest.raises(ValueError, match="Unsupported"):
        FileLoader.load(tmp_path / "x.txt")


def test_batch_over_demo_dataset(tmp_path):
    write_demo_dataset(tmp_path)
    (tmp_path / "broken.csv").write_text("not,a,recording\nx,y,z\n")
    progress = []
    results = BatchProcessor().process_directory(tmp_path, lambda *a: progress.append(a))

    by_name = {r["file_path"].split("/")[-1]: r for r in results}
    assert by_name["broken.csv"]["status"] == "error"
    assert by_name["nrem_sleep.csv"]["n_ripples"] >= DEMO_RECORDINGS["nrem_sleep"][2]
    assert progress[-1][:2] == (len(results), len(results))

    out = tmp_path / "out.txt"
    write_results_csv(results, out)
    assert out.read_text().count("\n") == len(results) + 1
