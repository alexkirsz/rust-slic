use image;
use js_sys;
use wasm_bindgen::prelude::*;

mod lbp;
mod slic;

use slic::{visualize, SLIC};

#[wasm_bindgen]
pub struct WasmSLIC {
    img: image::DynamicImage,
    slic: SLIC,
}

#[wasm_bindgen]
impl WasmSLIC {
    #[wasm_bindgen]
    pub fn new(img: &[u8], m: u32, s: u32, texture_coef: f32) -> Result<WasmSLIC, JsValue> {
        match image::load_from_memory(img) {
            Ok(img) => {
                let slic = SLIC::new(&img, m, s, texture_coef);
                Ok(WasmSLIC { img, slic })
            }
            Err(e) => Err(js_sys::Error::new(&e.to_string()).into()),
        }
    }

    #[wasm_bindgen]
    pub fn process(&self, err_threshold: f32, min_size: u32) -> Result<Vec<u8>, JsValue> {
        let regions = self.slic.process(err_threshold, min_size);
        let res = visualize(&self.img, &regions);

        let mut out = Vec::new();
        let encoder = image::png::PNGEncoder::new(&mut out);
        match encoder.encode(&res, res.width(), res.height(), image::ColorType::RGB(8)) {
            Err(e) => Err(js_sys::Error::new(&e.to_string()).into()),
            Ok(_) => Ok(out),
        }
    }
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
