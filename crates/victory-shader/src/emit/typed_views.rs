//! Host SRV component types and declared ranges for tracked guest descriptors.
use super::super::bitcode::Ty;
use super::Lower;
use crate::native::RootType;
use crate::resource::Source;

impl<'a> Lower<'a> {
    pub(super) fn scalar_uint_srv(&self, src: &Source, kind: u32) -> bool {
        if self.uint_srvs.is_empty() {
            return false;
        }
        let Source::Loaded { load, first } = src else {
            return false;
        };
        let Some(address) = self.scalar.get(load) else {
            return false;
        };
        let Some(root) = self.x.native.layout.get(address.param) else {
            return false;
        };
        if root.root != RootType::Table
            || address.dynamic.is_some()
            || !address.offset.is_multiple_of(4)
        {
            return false;
        }
        self.uint_srvs
            .contains(&(root.slot, address.offset / 4 + u32::from(*first), kind))
    }

    pub(super) fn resource_return_type(&mut self, uint: bool) -> Ty {
        if uint {
            self.s
                .m
                .named_struct("dx.types.ResRet.i32", vec![self.s.t.i32; 5])
        } else {
            self.s.t.res_ret
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{S_ENDPGM, lower_wave_with_views, root, texture_tables};
    use crate::emit::tiled_lighting_uint_srvs;

    #[test]
    fn fixed_uint_load_matches_only_the_named_root_field_and_dimension() {
        let vs = 0x8386ec5b_47a3b2de_49309239_7987b35du128.to_be_bytes();
        let views = tiled_lighting_uint_srvs(vs);
        let compile = |slot, offset: u32, dim: u32, views| {
            let words = [
                0xBE84_0300,
                0xBE85_0380, // s[4:5] = first root table
                0xF40C_0302,
                0xFA00_0000 | offset, // s_load_dwordx8 s[12:19]
                0xF000_1100 | dim,
                0x0003_0C00, // image_load v12, v[0:1], s[12:19]
                0xD834_0000,
                0x0000_0C00, // ds_write_b32 v0, v12
                S_ENDPGM,
            ];
            let (mut layout, mut used) = texture_tables();
            layout.truncate(1);
            used.truncate(1);
            layout[0] = root(layout[0].root, 0, slot);
            used[0].slot = slot;
            used[0].table_offset_dwords = offset / 4;
            lower_wave_with_views(&words, layout, used, false, views)
                .unwrap()
                .dxil
        };
        // Texture2D, root 1 + 48 dwords. Native immediate is in bytes.
        let matched = compile(1, 192, 8, views);
        assert_ne!(matched, compile(1, 192, 8, &[]));
        for (slot, offset, dim) in [(2, 192, 8), (1, 196, 8), (1, 192, 16)] {
            assert_eq!(
                compile(slot, offset, dim, views),
                compile(slot, offset, dim, &[])
            );
        }
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut matched.clone()).unwrap();
        }
    }
}
