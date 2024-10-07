#!/usr/bin/env just --justfile

release:
    cargo build --release

lint:
    cargo clippy

pack:
    wasm-pack build --target web

# Requires RustRover
file_dir := "http://localhost:8080"

browse:
    just _browse-{{os()}} 
    php -S localhost:8080

_browse-macos:
    open {{file_dir}}

_browse-linux:
    xdg-open {{file_dir}}

_browse-windows:
    start {{file_dir}}


web:
    just pack
    just browse
