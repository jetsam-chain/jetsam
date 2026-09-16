// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Build-time staging for the preflight-authenticated embedded `HistoryStep` packs.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use jetsam_ivc_core::proof::FieldShape;
use jetsam_miner::history_step_artifacts::{
    decode_history_step_runtime_metadata_pinned, history_step_matrix_file_name,
    HISTORY_STEP_PACK_LEAF_COUNT, HISTORY_STEP_PACK_VERSION_DIRECTORY,
    HISTORY_STEP_RUNTIME_METADATA_FILE, HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES,
};
use jetsam_recursive::acceptance::history_step_bank::CanonicalHistoryStepClassId;

const PACK_DIRECTORY_ENV: &str = "JETSAM_HISTORY_STEP_PACK_DIR";
const METADATA_DIGEST_ENV: &str = "JETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST";
/// The v1.3 pack, which governs blocks at and above the activation height.
///
/// Optional, and deliberately so: until the fork pack exists a release binary
/// still has to be buildable, deployable and able to speak to the live chain
/// with the pre-fork pack alone. A build without it simply cannot verify a
/// post-activation block — which, while the activation height is `None`,
/// is no block at all.
const PACK_DIRECTORY_V1_3_ENV: &str = "JETSAM_HISTORY_STEP_PACK_DIR_V1_3";
const METADATA_DIGEST_V1_3_ENV: &str =
    "JETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST_V1_3";

const GENERATED_FILE: &str = "history_step_pack.rs";
const STAGED_DIRECTORY: &str = "embedded-history-step";
const STAGED_DIRECTORY_V1_3: &str = "embedded-history-step-v1-3";
const MAX_COMPRESSED_LEAF_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CANONICAL_LEAF_BYTES: usize = 1024 * 1024 * 1024;
const ZSTD_WINDOW_LOG_MAX: u32 = 27;

#[derive(Clone, Copy)]
struct EmbeddedLeafSeal {
    shape: FieldShape,
    statement_digest: [u8; 32],
    canonical_bytes: usize,
}

struct EmbeddedLeaf {
    seal: EmbeddedLeafSeal,
}

fn main() {
    for variable in [
        PACK_DIRECTORY_ENV,
        METADATA_DIGEST_ENV,
        PACK_DIRECTORY_V1_3_ENV,
        METADATA_DIGEST_V1_3_ENV,
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let out_directory = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let generated_path = out_directory.join(GENERATED_FILE);

    // Two generations, one file. The pre-fork pack is what a release build
    // must always carry; the v1.3 pack is added when it exists.
    println!("cargo:rustc-check-cfg=cfg(has_pre_fork_pack)");
    let pre_fork = stage_generation(
        PACK_DIRECTORY_ENV,
        METADATA_DIGEST_ENV,
        STAGED_DIRECTORY,
        "GENERATED_HISTORY_STEP_PACK",
        &out_directory,
    );
    if pre_fork.is_none() {
        assert_ne!(
            env::var("PROFILE").as_deref(),
            Ok("release"),
            "release node builds require {PACK_DIRECTORY_ENV} and {METADATA_DIGEST_ENV}"
        );
    } else {
        // Lets the pack guard assert the real thing instead of returning
        // early: exactly one of its two forms compiles, so neither build can
        // pass it without checking something.
        println!("cargo:rustc-cfg=has_pre_fork_pack");
    }
    let post_fork = stage_generation(
        PACK_DIRECTORY_V1_3_ENV,
        METADATA_DIGEST_V1_3_ENV,
        STAGED_DIRECTORY_V1_3,
        "GENERATED_HISTORY_STEP_PACK_V1_3",
        &out_directory,
    );

    let mut generated = String::new();
    generated.push_str(
        &pre_fork.unwrap_or_else(|| empty_generation("GENERATED_HISTORY_STEP_PACK")),
    );
    generated.push_str(
        &post_fork.unwrap_or_else(|| empty_generation("GENERATED_HISTORY_STEP_PACK_V1_3")),
    );
    write_if_changed(&generated_path, generated.as_bytes());
}

fn empty_generation(static_name: &str) -> String {
    format!("static {static_name}: Option<EmbeddedHistoryStepPack> = None;\n")
}

/// Stage one pack generation, or `None` when its two variables are unset.
fn stage_generation(
    directory_variable: &str,
    digest_variable: &str,
    staged_directory: &str,
    static_name: &str,
    out_directory: &Path,
) -> Option<String> {
    match (env::var_os(directory_variable), env::var_os(digest_variable)) {
        (None, None) => None,
        (Some(pack_directory), Some(metadata_digest)) => {
            let metadata_digest = parse_hex_digest(
                &metadata_digest
                    .into_string()
                    .unwrap_or_else(|_| panic!("{digest_variable} is not UTF-8")),
                digest_variable,
            );
            Some(embed_release_pack(
                &PathBuf::from(pack_directory),
                metadata_digest,
                out_directory,
                staged_directory,
                static_name,
            ))
        }
        _ => panic!("{directory_variable} and {digest_variable} must be set together"),
    }
}

fn embed_release_pack(
    pack_root: &Path,
    metadata_digest: [u8; 32],
    out_directory: &Path,
    staged_directory_name: &str,
    static_name: &str,
) -> String {
    let version_directory = pack_root.join(HISTORY_STEP_PACK_VERSION_DIRECTORY);
    let metadata_path = version_directory.join(HISTORY_STEP_RUNTIME_METADATA_FILE);
    let metadata = read_bounded(
        &metadata_path,
        HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES as u64,
    );
    let runtime_metadata = decode_history_step_runtime_metadata_pinned(&metadata, metadata_digest)
        .unwrap_or_else(|error| {
            panic!(
                "validate pinned HistoryStep runtime metadata {}: {error}",
                metadata_path.display()
            )
        });
    println!("cargo:rerun-if-changed={}", metadata_path.display());

    let staged_directory = out_directory.join(staged_directory_name);
    fs::create_dir_all(&staged_directory).unwrap_or_else(|error| {
        panic!(
            "create embedded HistoryStep staging directory {}: {error}",
            staged_directory.display()
        )
    });
    write_if_changed(
        &staged_directory.join(HISTORY_STEP_RUNTIME_METADATA_FILE),
        &metadata,
    );

    let mut build_leaves = Vec::with_capacity(HISTORY_STEP_PACK_LEAF_COUNT);
    for index in 0..HISTORY_STEP_PACK_LEAF_COUNT {
        let class = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        let leaf_path = version_directory.join(history_step_matrix_file_name(class));
        let compressed = read_bounded(&leaf_path, MAX_COMPRESSED_LEAF_BYTES);
        let entry = runtime_metadata.bank().entry(class);
        // Pack generation/preflight authenticates the relation once. Building
        // an executable only stages those approved bytes and records the
        // decoded-size bound required by runtime decompression.
        let leaf = stage_leaf(
            &compressed,
            entry.shape(),
            entry.matrix_digest(),
            &leaf_path,
        );
        write_if_changed(
            &staged_directory.join(history_step_matrix_file_name(class)),
            &compressed,
        );
        build_leaves.push(leaf);
        println!("cargo:rerun-if-changed={}", leaf_path.display());
    }
    let build_leaves: [EmbeddedLeaf; HISTORY_STEP_PACK_LEAF_COUNT] = build_leaves
        .try_into()
        .unwrap_or_else(|_| unreachable!("one build result per HistoryStep class"));
    render_generated_pack(
        metadata_digest,
        &build_leaves,
        staged_directory_name,
        static_name,
    )
}

fn stage_leaf(
    compressed: &[u8],
    shape: FieldShape,
    statement_digest: [u8; 32],
    path: &Path,
) -> EmbeddedLeaf {
    let mut decoder = zstd::stream::read::Decoder::new(compressed)
        .unwrap_or_else(|error| panic!("open compressed matrix {}: {error}", path.display()));
    decoder
        .window_log_max(ZSTD_WINDOW_LOG_MAX)
        .unwrap_or_else(|error| panic!("bound zstd window for {}: {error}", path.display()));
    let canonical_bytes = std::io::copy(
        &mut decoder.take(MAX_CANONICAL_LEAF_BYTES as u64 + 1),
        &mut std::io::sink(),
    )
    .unwrap_or_else(|error| panic!("measure decoded matrix {}: {error}", path.display()));
    assert!(
        canonical_bytes <= MAX_CANONICAL_LEAF_BYTES as u64,
        "HistoryStep matrix {} exceeds the canonical size bound",
        path.display()
    );
    EmbeddedLeaf {
        seal: EmbeddedLeafSeal {
            shape,
            statement_digest,
            canonical_bytes: usize::try_from(canonical_bytes)
                .expect("bounded canonical matrix length fits usize"),
        },
    }
}

fn read_bounded(path: &Path, max_bytes: u64) -> Vec<u8> {
    let metadata = fs::metadata(path)
        .unwrap_or_else(|error| panic!("inspect release artifact {}: {error}", path.display()));
    assert!(
        metadata.is_file() && metadata.len() <= max_bytes,
        "release artifact {} is not a bounded regular file",
        path.display()
    );
    fs::read(path)
        .unwrap_or_else(|error| panic!("read release artifact {}: {error}", path.display()))
}

fn parse_hex_digest(encoded: &str, variable: &str) -> [u8; 32] {
    assert_eq!(
        encoded.len(),
        64,
        "{variable} must be exactly 64 lowercase hexadecimal characters"
    );
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let high = decode_lower_hex(encoded.as_bytes()[index * 2])
            .unwrap_or_else(|| panic!("{variable} is not lowercase hexadecimal"));
        let low = decode_lower_hex(encoded.as_bytes()[index * 2 + 1])
            .unwrap_or_else(|| panic!("{variable} is not lowercase hexadecimal"));
        *byte = (high << 4) | low;
    }
    digest
}

const fn decode_lower_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn render_generated_pack(
    metadata_digest: [u8; 32],
    leaves: &[EmbeddedLeaf; HISTORY_STEP_PACK_LEAF_COUNT],
    staged_directory_name: &str,
    static_name: &str,
) -> String {
    let mut generated = format!(
        "static {static_name}: Option<EmbeddedHistoryStepPack> =\n\
         Some(EmbeddedHistoryStepPack {{\n"
    );
    writeln!(
        &mut generated,
        "    runtime_metadata: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{staged_directory_name}/{HISTORY_STEP_RUNTIME_METADATA_FILE}\")),"
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut generated,
        "    runtime_metadata_digest: {},",
        render_digest(metadata_digest)
    )
    .expect("writing to String cannot fail");
    generated.push_str("    leaves: [\n");
    for (index, leaf) in leaves.iter().enumerate() {
        let class = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        let seal = leaf.seal;
        writeln!(
            &mut generated,
            "        unsafe {{ jetsam_miner::EmbeddedHistoryStepMatrixLeaf::from_release_build(\n            jetsam_recursive::acceptance::history_step_bank::CanonicalHistoryStepClassId::from_index({index}).unwrap(),\n            include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{staged_directory_name}/{}\")),\n            jetsam_ivc_core::field_r1cs::BuildAuthenticatedFieldR1csSeal::from_release_build(\n                jetsam_ivc_core::proof::FieldShape {{ m: {}, k_log: {}, k_skip: {}, const_pin: {} }},\n                {},\n                {},\n            ),\n        ) }},",
            history_step_matrix_file_name(class),
            seal.shape.m,
            seal.shape.k_log,
            seal.shape.k_skip,
            render_const_pin(seal.shape.const_pin),
            render_digest(seal.statement_digest),
            seal.canonical_bytes,
        )
        .expect("writing to String cannot fail");
    }
    generated.push_str("    ],\n});\n");
    generated
}

fn render_const_pin(pin: Option<usize>) -> String {
    pin.map_or_else(|| "None".to_owned(), |column| format!("Some({column})"))
}

fn render_digest(digest: [u8; 32]) -> String {
    let mut rendered = String::from("[");
    for (index, byte) in digest.iter().enumerate() {
        if index != 0 {
            rendered.push_str(", ");
        }
        write!(&mut rendered, "0x{byte:02x}").expect("writing to String cannot fail");
    }
    rendered.push(']');
    rendered
}

fn write_if_changed(path: &Path, bytes: &[u8]) {
    if matches!(fs::read(path), Ok(existing) if existing == bytes) {
        return;
    }
    fs::write(path, bytes)
        .unwrap_or_else(|error| panic!("write generated artifact {}: {error}", path.display()));
}
