import zarr
minian_dir = r'H:\data\vsekulic\OF_test\G15-ST721-hM4D\2022_01_11-TFC_cond\18_10_56-TFC_cond\Miniscope\minian_crossreg1_crossreg2_crossreg4_crossreg6_crossreg7'

a_zarr = zarr.load(os.path.join(minian_dir, "A.zarr"))
s_zarr = zarr.load(os.path.join(minian_dir, "S.zarr"))
c_zarr = zarr.load(os.path.join(minian_dir, "C.zarr"))

print("A.zarr:", a_zarr['A'].shape, "unit_ids:", len(a_zarr['unit_id']))
print("S.zarr:", s_zarr['S'].shape, "unit_ids:", len(s_zarr['unit_id']))
print("C.zarr:", c_zarr['C'].shape, "unit_ids:", len(c_zarr['unit_id']))

# Check overlap
a_ids = set(a_zarr['unit_id'])
s_ids = set(s_zarr['unit_id'])
print(f"\nIn S but not A ({len(s_ids - a_ids)}): {sorted(s_ids - a_ids)}")
print(f"In A but not S ({len(a_ids - s_ids)}): {sorted(a_ids - s_ids)}")