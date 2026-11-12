"""Build the single README and executed Notebook from prepared data."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'result'


def read_csv(name):
    with (RESULT/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def column(rows,key):
    return np.array([float(r[key]) for r in rows])


def main():
    s=json.loads((RESULT/'summary.json').read_text(encoding='utf-8'))
    checks=json.loads((RESULT/'verification.json').read_text(encoding='utf-8'))
    if not checks['passed']:
        raise RuntimeError('Processing checks have not passed')
    g=read_csv('gps.csv'); a=read_csv('temperature_alignment.csv'); p=read_csv('gp_samples.csv')
    matched=[r for r in a if r['alignment_status']=='matched']
    figs=RESULT/'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ax=axes[0]
    ax.plot(column(g,'x_east_m'),column(g,'y_north_m'),color='.65',lw=.8,label='Recorded GPS')
    scatter=ax.scatter(column(p,'x_east_m'),column(p,'y_north_m'),c=column(p,'temperature_c'),
                       cmap='viridis',s=9,zorder=2)
    ax.scatter([0],[0],marker='*',s=100,c='black',label='ENU origin',zorder=3)
    ax.set(xlabel='East (m)',ylabel='North (m)',title='Temperature assigned along the measured route')
    ax.set_aspect('equal'); ax.legend(loc='lower left',fontsize=8)
    fig.colorbar(scatter,ax=ax,label='HOBO temperature (°C)',shrink=.85)
    ax=axes[1]
    origin=s['coordinate_system']['time_origin_utc_ns']
    t=np.array([(int(r['timestamp_utc_ns'])-origin)/1e9 for r in a])/60
    ax.plot(t,column(a,'temperature_c'),color='.55',lw=1,label='All HOBO readings')
    ax.scatter(column(p,'t_s')/60,column(p,'temperature_c'),s=5,color='#168478',label='Matched readings')
    ax.axvspan(0,(int(g[-1]['header_timestamp_ns'])-origin)/6e10,color='#168478',alpha=.08,label='GPS support')
    ax.set(xlabel='Minutes since first GPS fix',ylabel='Temperature (°C)',title='Recorded temperature over time')
    ax.legend(fontsize=8)
    fig.suptitle('2026-11-03 field recording | zero additional clock offset assumed',fontsize=13)
    fig.savefig(figs/'trajectory_temperature.png',dpi=160); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    axes[0].hist(column(matched,'nearest_gps_time_distance_s'),bins=25,color='#168478',alpha=.9)
    axes[0].set(xlabel='Distance to nearest GPS timestamp (s)',ylabel='Temperature records',
                title='Sampling phase difference; not measured clock error')
    st=column(matched,'t_s')/60
    axes[1].plot(st,column(matched,'gps_bracket_gap_s'),color='#2d5b92',lw=1)
    axes[1].axhline(s['assumptions']['max_gps_gap_s'],color='#a33737',ls='--',label='Maximum allowed gap')
    axes[1].set(xlabel='Minutes since first GPS fix',ylabel='GPS bracket interval (s)',
                title='Interpolation support')
    axes[1].legend(fontsize=8)
    fig.savefig(figs/'alignment_diagnostics.png',dpi=160); plt.close(fig)

    md=f'''# Real Data Integration — Step 1: Temperature and GPS Alignment

## Purpose and dataset

Prepare traceable `(x, y, t, T)` observations from Julian's **2026-11-03** lawnmower recording near the baseball field. This stage covers timestamp conversion, GPS decoding, local metric coordinates and data-quality checks. GP training and ROS 2 navigation are subsequent stages.

**Result:** {s['matched_count']} temperature observations have interpolated GPS positions under an **unverified zero additional clock-offset assumption**. Processing checks passed; physical sensor synchronization is still unresolved.

## Inputs and dependencies

Original inputs remain in `../../../Data/Real_spatialtemporal_data/` relative to this folder:

- `lawnmower_5m_test_2026-11-03 14_41_39 EST (Data EST).xlsx`: HOBO temperature, `Data!A1:C1101`; `Events` and `Details` preserve logging metadata.
- `2026-11-03-welle-baseball-lawnmower/nov-3-baseball-1/nov-3-baseball-1_0.db3` and `metadata.yaml`: ROS recording. Only `/calian_gnss/gps` (`sensor_msgs/msg/NavSatFix`) is decoded.

Processing uses Python, NumPy and openpyxl for read-only Excel access, plus ROS 2 Jazzy `rclpy` and `sensor_msgs` in the existing `bunker-jazzy` Docker container. Selected GPS messages are streamed to ROS through stdin, so the 25.4 GB bag need not be copied or mounted. Report generation uses Matplotlib, nbformat, nbclient and an installed Python Jupyter kernel. See [requirements.txt](testing_code/requirements.txt).

## Method and assumptions

1. Convert the workbook's explicitly labelled **EST (UTC−05:00)** timestamps to UTC. Preserve original timestamps, sample IDs and Excel row numbers. Use the GPS **header** timestamp for position matching; retain bag receipt time separately.
2. Convert WGS84 latitude/longitude/altitude to ECEF and then a fixed local **ENU** frame. Origin: the first valid GPS fix, latitude `{s['coordinate_system']['origin_latitude_deg']}`, longitude `{s['coordinate_system']['origin_longitude_deg']}`, altitude `{s['coordinate_system']['origin_altitude_m']} m`. `x` is east and `y` is north, in metres. `t_s` is seconds since `{s['coordinate_system']['time_origin_utc']}`.
3. Interpolate ENU position between consecutive valid GPS fixes at each temperature timestamp. Require valid finite positions and a bracket interval ≤{s['assumptions']['max_gps_gap_s']:g} s. This gap threshold is a processing choice, not a calibrated accuracy bound. Do not extrapolate, bridge an invalid fix or invent a position for rejected records.
4. Retain all temperature rows with match/exclusion reasons. Temperature values are not interpolated, smoothed or detrended. Preserve decoded GPS status, covariance and message IDs for review.

The approximately 0.325 s nearest-sample separation is **sampling phase**, not evidence of clock bias. No common HOBO/GPS synchronization event was identified. The script accepts an explicit `--temperature-offset-s` (positive adds time to HOBO), but does not estimate or certify it. Sensor response delay and the HOBO-to-GPS mounting offset are also unknown. Local ENU is not yet registered to a ROS `map` frame.

## Results

| Measure | Result |
|---|---:|
| HOBO records / interval | {s['temperature_count']} / 1 s |
| Valid GPS records | {s['valid_gps_count']} of {s['gps_count']} |
| Matched temperature records | {s['matched_count']} |
| Excluded before / after GPS support | {s['alignment_counts'].get('before_gps',0)} / {s['alignment_counts'].get('after_gps',0)} |
| Matched local interval (EST) | 14:24:16–14:40:35 |
| Full recorded temperature range | {s['temperature_c']['min']:.2f}–{s['temperature_c']['max']:.2f} °C |
| GPS extent, east × north | {s['gps_extent_east_north_m'][0]:.2f} × {s['gps_extent_east_north_m'][1]:.2f} m |
| Sum of consecutive horizontal GPS distances | {s['horizontal_path_length_m']:.2f} m |
| Largest GPS interval | {s['gps_interval_s']['max']:.3f} s |
| Median nearest GPS time distance | {s['nearest_gps_time_distance_s']['median']:.3f} s |
| Automated processing checks | {checks['tests_run']} passed |

The recorded trajectory resembles the supplied lawnmower route. Pass spacing is approximately 5 m according to the collector; this pipeline does not treat that statement as a surveyed calibration. GPS status 2 denotes GBAS and does not by itself establish an RTK fixed solution or centimetre accuracy. A single moving run cannot isolate spatial variation from temporal drift or validate multi-day prediction. The data describe sampled locations, not occupancy or traversability.

## Files and data contract

| File | Role |
|---|---|
| [test_report.ipynb](test_report.ipynb) | Executed summary, sample preview, figures, clock sensitivity and conclusions |
| [result/gp_samples.csv](result/gp_samples.csv) | Matched observations for the next GP stage; `x_east_m`, `y_north_m`, `t_s`, `temperature_c` plus source identity and clock flag |
| [result/temperature_alignment.csv](result/temperature_alignment.csv) | All original temperature rows; match status, source and UTC timestamps, bracketing GPS message IDs/times, interpolation fraction and gap |
| [result/gps.csv](result/gps.csv) | Decoded GPS headers/receipt times, WGS84 coordinates, ENU coordinates, status, covariance and validity |
| [result/summary.json](result/summary.json) | Origin, units, assumptions, source topic inventory, logging metadata and timing sensitivity |
| [result/manifest.json](result/manifest.json) | Full SHA-256 of original Excel/bag/metadata, selected CDR digest, script/runtime and derived-file hashes |
| [result/verification.json](result/verification.json) | Executed processing checks and their scope |

`timestamp_utc_ns` and GPS timestamps are integer Unix nanoseconds; keep them as integers, not spreadsheet floating-point timestamps. Covariance entries `cov_0_m2` through `cov_8_m2` retain the message's row-major ENU position covariance. Blank position fields mean an excluded temperature row, never a zero position. Every prepared row keeps `clock_alignment_verified=False`. For additional runs, reuse a common spatial/time origin before combining them; this single-run default resets to the first valid fix.

## Reproduce

From this folder, with Docker Desktop and `bunker-jazzy` running:

```powershell
powershell -ExecutionPolicy Bypass -File .\\testing_code\\Run-Alignment.ps1
```

The launcher uses the existing managed extraction Python and reporting environment. Override `-DataPython` and `-ReportPython` for other installations. It regenerates derived results, checks, this README and the single Notebook. For a ROS-sourced Linux Python, run `prepare_data.py --ros-python /usr/bin/python3` with a host Python that has NumPy/openpyxl. Source files are opened read-only. Hashing the complete bag records provenance, not a full bag integrity certification. No Git commit or upload is performed.

Next: confirm HOBO/GPS clock setup or a common event, and sensor response/mounting details. Then use blocked held-out route/time data for initial GP validation. Until synchronization is confirmed, treat the prepared positions as provisional and retain the documented assumption in any experiment.
'''
    (ROOT/'README.md').write_text(md,encoding='utf-8')
    nb=nbformat.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                 'language_info':{'name':'python'},'description':'Executed data preparation report; no GP training or navigation.'}
    cells=[nbformat.v4.new_markdown_cell('''# Real Data Integration: Temperature and GPS Alignment

**Purpose:** prepare traceable field observations for subsequent GP tests. This report reads the saved processing outputs. Original data, assumptions and reproduction instructions are documented in the [README](README.md).

**Clock alignment remains unverified.** EST is converted to UTC and zero additional sensor-clock offset is assumed. Numerical processing checks do not establish physical synchronization.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import csv, json, hashlib, html
from IPython.display import display, HTML, Image
root=Path.cwd()
if not (root/'result'/'summary.json').exists():
    raise RuntimeError('Run this notebook from the Real_Data_Integration folder.')
result=root/'result'
s=json.loads((result/'summary.json').read_text(encoding='utf-8'))
v=json.loads((result/'verification.json').read_text(encoding='utf-8'))
with (result/'gp_samples.csv').open(encoding='utf-8',newline='') as f:
    samples=list(csv.DictReader(f))
def table(headers, rows):
    def row(values,tag):
        return '<tr>'+''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in values)+'</tr>'
    display(HTML('<table>'+row(headers,'th')+''.join(row(r,'td') for r in rows)+'</table>'))
table(['Measure','Result'],[
    ['Run',s['run_id']],['Temperature / valid GPS',f"{s['temperature_count']} / {s['valid_gps_count']}"],
    ['Matched / before / after',f"{s['matched_count']} / {s['alignment_counts'].get('before_gps',0)} / {s['alignment_counts'].get('after_gps',0)}"],
    ['Matched UTC interval',s['matched_start_utc']+' to '+s['matched_end_utc']],
    ['Extent east × north (m)',str([round(x,2) for x in s['gps_extent_east_north_m']])],
    ['Recorded horizontal path (m)',round(s['horizontal_path_length_m'],2)],
    ['Processing checks passed',v['tests_run'] if v['passed'] else 'FAILED'],
    ['Physical clock alignment verified',s['assumptions']['clock_alignment_verified']]])
fields=['sample_id','timestamp_utc','t_s','x_east_m','y_north_m','temperature_c']
table(fields,[[r[k] if k in fields[:2] else f'{float(r[k]):.4f}' for k in fields] for r in samples[:3]+samples[-3:]])'''),
    nbformat.v4.new_markdown_cell('''## Measured trajectory and temperature

Colour represents temperature assigned at the assumed synchronized timestamp. The trajectory is in local ENU metres. This is a plot of recorded samples, not a reconstructed temperature field or navigation map.'''),
    nbformat.v4.new_code_cell("display(Image(filename=str(result/'figures'/'trajectory_temperature.png'), embed=True))\ndisplay(Image(filename=str(result/'figures'/'alignment_diagnostics.png'), embed=True))"),
    nbformat.v4.new_markdown_cell('''## Sensitivity and validation

The following diagnostic shifts temperature timestamps by hypothetical offsets and recomputes positions. It shows the effect of timing uncertainty; it neither estimates the true offset nor supplies a calibrated error bound. Nearest-sample time distance and bag receipt latency cannot verify the HOBO clock.'''),
    nbformat.v4.new_code_cell('''table(['Hypothetical extra offset (s)','Comparable rows','Median position change (m)','Maximum change (m)'],
      [[r['assumed_extra_offset_s'],r['comparable_samples'],
        round(r['horizontal_displacement_m']['median'],3),round(r['horizontal_displacement_m']['max'],3)]
       for r in s['offset_sensitivity']])
m=json.loads((result/'manifest.json').read_text(encoding='utf-8'))
for name,expected in m['outputs'].items():
    assert hashlib.sha256((result/name).read_bytes()).hexdigest()==expected, name
assert v['passed'] and len(samples)==s['matched_count']
assert all(r['clock_alignment_verified']=='False' for r in samples)
print('All saved data-product hashes match. All matched rows retain the unverified-clock flag.')
print('Executed checks:')
print('\\n'.join(v['test_names']))'''),
    nbformat.v4.new_markdown_cell(f'''## Conclusion

- **{s['matched_count']} matched observations** are available for initial GP data work under the recorded clock assumption. The other {s['temperature_count']-s['matched_count']} temperatures remain in the alignment audit with exclusion reasons.
- Recorded geometry spans approximately **{s['gps_extent_east_north_m'][0]:.1f} × {s['gps_extent_east_north_m'][1]:.1f} m**. Do not rescale these physical coordinates to an earlier synthetic map.
- Clock offset, thermal response delay and the sensor mounting offset remain unverified. Confirm the clock procedure/common event before treating spatial associations as validated.
- One moving run does not separate temporal drift from spatial structure. No kernel fitting, policy ranking, field reconstruction or ROS navigation was executed.
- Next: resolve timing evidence and run an initial GP evaluation with held-out route/time blocks. A map-frame registration and navigable map will be needed separately for physical navigation.''')]
    nb.cells=cells
    client=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.validate(nb)
    nbformat.write(nb,ROOT/'test_report.ipynb')
    print('Wrote one README, one executed Notebook, and two figures.')


if __name__=='__main__':
    main()
