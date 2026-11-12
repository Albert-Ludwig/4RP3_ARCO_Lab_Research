"""Check clock conversion, interpolation boundaries, coordinates and real rows."""
import csv
import json
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np
import openpyxl
from prepare_data import (EXPERIMENT, REPOSITORY, bracket, digest,
                          enu_from_geodetic, iso_ns, utc_ns, write_json)


def read_csv(name):
    with (EXPERIMENT/'result'/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


class AlignmentTests(unittest.TestCase):
    def test_est_conversion_and_offset_sign(self):
        t = datetime(1970,1,1)
        self.assertEqual(utc_ns(t), 18000*10**9)
        self.assertEqual(utc_ns(t,2), 18002*10**9)
        self.assertEqual(iso_ns(utc_ns(datetime(2026,11,3,14,23,13))),
                         '2026-11-03T19:23:13.000000000Z')

    def test_nanosecond_precision_and_exact_boundaries(self):
        start=1793733855325545540
        times=np.array([start,start+10**9],dtype=np.int64)
        valid=np.array([True,True])
        self.assertEqual(bracket(times,start-1,valid,2)[1],'before_gps')
        self.assertEqual(bracket(times,start+10**9+1,valid,2)[1],'after_gps')
        self.assertEqual(bracket(times,start,valid,2)[0][:3],(0,0,0.0))
        self.assertEqual(bracket(times,start+10**9,valid,2)[0][:3],(1,1,0.0))
        self.assertEqual(bracket(times,start+250000000,valid,2)[0][2],.25)

    def test_invalid_fix_and_gap_are_not_bridged(self):
        times=np.array([0,10**9,5*10**9],dtype=np.int64)
        self.assertEqual(bracket(times,5*10**8,np.array([True,False,True]),2)[1],
                         'invalid_gps_endpoint')
        self.assertEqual(bracket(times,2*10**9,np.ones(3,dtype=bool),2)[1],'gps_gap_too_large')

    def test_enu_against_analytic_equator_geometry(self):
        # Independent circle geometry at WGS84 equator, including vertical axis.
        a=6378137.0
        angle=np.deg2rad(.001)
        actual=enu_from_geodetic([[0,.001,0],[0,0,10]],np.array([0,0,0]))
        expected=[[a*np.sin(angle),0,a*(np.cos(angle)-1)],[0,0,10]]
        np.testing.assert_allclose(actual,expected,atol=1e-8)
        origin=np.array([43.265474,-79.928666,45.3434])
        np.testing.assert_allclose(enu_from_geodetic([origin],origin),0,atol=1e-10)

    def test_actual_counts_and_source_temperature_preservation(self):
        result=EXPERIMENT/'result'
        summary=json.loads((result/'summary.json').read_text(encoding='utf-8'))
        manifest=json.loads((result/'manifest.json').read_text(encoding='utf-8'))
        xlsx=next(REPOSITORY/p['path'] for p in manifest['inputs'] if p['path'].endswith('.xlsx'))
        w=openpyxl.load_workbook(xlsx,read_only=True,data_only=True)
        source=list(w['Data'].values)[1:]; w.close()
        rows=read_csv('temperature_alignment.csv'); samples=read_csv('gp_samples.csv')
        self.assertEqual(len(rows),len(source))
        self.assertEqual(len(samples),summary['matched_count'])
        self.assertEqual(sum(summary['alignment_counts'].values()),len(source))
        for raw,row in zip(source,rows):
            self.assertEqual(int(row['sample_id']),raw[0])
            self.assertEqual(row['source_local_time'],raw[1].isoformat())
            self.assertEqual(float(row['temperature_c']),raw[2])
            self.assertEqual(row['clock_alignment_verified'],'False')
        accepted={r['sample_id'] for r in rows if r['alignment_status']=='matched'}
        self.assertEqual(accepted,{r['sample_id'] for r in samples})
        for r in rows:
            if r['alignment_status']!='matched':
                self.assertEqual(r['x_east_m'],'')
                self.assertEqual(r['y_north_m'],'')

    def test_actual_interpolation_against_numpy_and_bracket_provenance(self):
        gps=read_csv('gps.csv'); samples=read_csv('gp_samples.csv')
        rows=[r for r in read_csv('temperature_alignment.csv') if r['alignment_status']=='matched']
        t0=int(gps[0]['header_timestamp_ns'])
        gt=np.array([(int(r['header_timestamp_ns'])-t0)/1e9 for r in gps])
        st=np.array([(int(r['timestamp_utc_ns'])-t0)/1e9 for r in samples])
        for field in ('x_east_m','y_north_m'):
            expected=np.interp(st,gt,[float(r[field]) for r in gps])
            np.testing.assert_allclose([float(r[field]) for r in samples],expected,atol=1e-9,rtol=0)
        ids={r['message_id']:r for r in gps}
        for row in rows:
            t=int(row['timestamp_utc_ns'])
            left=ids[row['gps_left_message_id']]; right=ids[row['gps_right_message_id']]
            self.assertLessEqual(int(left['header_timestamp_ns']),t)
            self.assertLessEqual(t,int(right['header_timestamp_ns']))
            self.assertTrue(0<=float(row['interpolation_fraction'])<=1)
            self.assertLessEqual(float(row['gps_bracket_gap_s']),2.)
            self.assertEqual(row['gps_left_utc_ns'],left['header_timestamp_ns'])
            self.assertEqual(row['gps_right_utc_ns'],right['header_timestamp_ns'])

    def test_actual_coordinate_scale_against_local_ellipsoid_differential(self):
        gps=read_csv('gps.csv')
        lat=np.deg2rad([float(r['latitude_deg']) for r in gps])
        lon=np.deg2rad([float(r['longitude_deg']) for r in gps])
        e2=6.6943799901413165e-3; a=6378137.
        n=a/np.sqrt(1-e2*np.sin(lat[0])**2)
        m=a*(1-e2)/(1-e2*np.sin(lat[0])**2)**1.5
        h=float(gps[0]['altitude_m'])
        x=(lon-lon[0])*(n+h)*np.cos(lat[0]); y=(lat-lat[0])*(m+h)
        np.testing.assert_allclose([float(r['x_east_m']) for r in gps],x,atol=.003,rtol=0)
        np.testing.assert_allclose([float(r['y_north_m']) for r in gps],y,atol=.003,rtol=0)

    def test_output_hashes_and_unverified_clock_flag(self):
        result=EXPERIMENT/'result'
        manifest=json.loads((result/'manifest.json').read_text(encoding='utf-8'))
        summary=json.loads((result/'summary.json').read_text(encoding='utf-8'))
        for name,sha in manifest['outputs'].items():
            self.assertEqual(digest(result/name),sha,name)
        self.assertTrue(manifest['source_size_and_mtime_unchanged'])
        self.assertFalse(summary['assumptions']['clock_alignment_verified'])
        self.assertEqual(summary['coordinate_system']['ros_map_transform'],'not established')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AlignmentTests)
    names=[t.id().split('.')[-1] for t in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(EXPERIMENT/'result'/'verification.json',dict(
        passed=result.wasSuccessful(), tests_run=result.testsRun, test_names=names,
        failures=[str(t) for t,_ in result.failures],errors=[str(t) for t,_ in result.errors],
        scope='Processing and consistency checks only. Independent sensor clock synchronization remains unverified.'))
    raise SystemExit(not result.wasSuccessful())
