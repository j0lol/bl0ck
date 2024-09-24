#!/usr/bin/env just --justfile

release:
    cargo build --release

lint:
    cargo clippy

pack:
    wasm-pack build --target web

# Requires RustRover
file_dir := "http://localhost:63342/bl0ck/index.html?_ijt=6adftqhk5fvj3fik73h0ri92c0&_ij_reload=RELOAD_ON_SAVE"
browse:
    just _browse-{{os()}}

_browse-macos:
    open {{file_dir}}

_browse-linux:
    xdg-open {{file_dir}}

_browse-windows:
    start {{file_dir}}
