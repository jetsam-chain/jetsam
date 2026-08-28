// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

use iced::window;

const APPLICATION_ICON: &[u8] = include_bytes!("../assets/app-icons/Jetsam-64.png");

pub fn icon() -> window::Icon {
    let icon = image::load_from_memory(APPLICATION_ICON)
        .expect("embedded application icon is valid")
        .into_rgba8();
    let (width, height) = icon.dimensions();
    window::icon::from_rgba(icon.into_raw(), width, height)
        .expect("embedded application icon dimensions are valid")
}
