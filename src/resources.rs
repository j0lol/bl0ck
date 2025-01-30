

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
	let window = web_sys::window().unwrap();
	let location = window.location();
	let mut origin = location.origin().unwrap();
	if !origin.ends_with("learn-wgpu") {
		origin = format!("{}/learn-wgpu", origin);
	}
	let base = reqwest::Url::parse(&format!("{}/", origin)).unwrap();
	base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> Result<String, Box<dyn Error>> {
	cfg_if! {
		if #[c]
	}
}