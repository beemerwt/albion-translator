mod cli;
pub mod translation;
pub mod translator;

use albion_network_lib::{
    ChatMessage, DecodedPacket, EventCode, ExtractedPacket, HostFilter, PhotonParser,
    PhotonParserConfig, extract_udp_payload,
};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Local, TimeZone};
use clap::Parser;
use cli::Args;
use pcap::{Active, Capture, Device};
use serde_json::Value;
use std::{
    borrow::Cow,
    io, mem,
    path::Path,
    ptr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

const CAPTURE_FILTER: &str = "udp port 5056 or udp port 4535";
const HOSTS_PATH: &str = "hosts.txt";
const MAX_TRANSLATION_EXPANSION_FACTOR: usize = 4;
const MIN_TRANSLATION_MAX_CHARS: usize = 120;
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
    let args = Args::parse();

    let host_filter = HostFilter::from_file(Path::new(HOSTS_PATH))
        .map_err(|error| anyhow!(error.0))
        .with_context(|| format!("failed to load CIDR ranges from {HOSTS_PATH}"))?;
    let translator_server = if args.dry_run {
        None
    } else {
        Some(translator::TranslatorServer::start().context("failed to start translator sidecar")?)
    };

    run_capture(&host_filter, &args, translator_server.as_ref())
}

fn run_capture(
    host_filter: &HostFilter,
    args: &Args,
    translator_server: Option<&translator::TranslatorServer>,
) -> Result<()> {
    install_shutdown_handlers().context("failed to install shutdown signal handlers")?;

    let mut captures = open_captures()?;

    let mode = if args.dry_run { "dry run; " } else { "" };
    eprintln!(
        "capturing on {} interfaces with filter {:?} and {HOSTS_PATH}; {mode}press Ctrl-C to stop",
        captures.len(),
        CAPTURE_FILTER
    );

    let config = PhotonParserConfig::with_defaults("live".to_string(), args.debug);
    let mut parser = PhotonParser::new(config);
    let mut packet_number = 0usize;

    while !SHUTDOWN_REQUESTED.load(Ordering::SeqCst) {
        let mut received_packet = false;

        for source in &mut captures {
            if SHUTDOWN_REQUESTED.load(Ordering::SeqCst) {
                break;
            }

            match source.capture.next_packet() {
                Ok(packet) => {
                    received_packet = true;
                    packet_number += 1;
                    handle_packet(
                        &mut parser,
                        packet_number,
                        packet.header.ts.tv_sec,
                        packet.data,
                        source.link_type,
                        host_filter,
                        args,
                        translator_server,
                    )
                    .with_context(|| format!("capture source {}", source.name))?;
                }
                Err(pcap::Error::NoMorePackets | pcap::Error::TimeoutExpired) => {}
                Err(error) => {
                    eprintln!("warning: capture on {} failed: {error}", source.name);
                }
            }
        }

        if !received_packet {
            thread::sleep(Duration::from_millis(10));
        }
    }

    eprintln!("stopped after {packet_number} packets");
    Ok(())
}

extern "C" fn request_shutdown(_: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

fn install_shutdown_handlers() -> io::Result<()> {
    unsafe {
        let mut action: libc::sigaction = mem::zeroed();
        action.sa_sigaction = request_shutdown as *const () as usize;
        action.sa_flags = 0;
        libc::sigemptyset(&mut action.sa_mask);

        if libc::sigaction(libc::SIGINT, &action, ptr::null_mut()) != 0 {
            return Err(io::Error::last_os_error());
        }

        if libc::sigaction(libc::SIGTERM, &action, ptr::null_mut()) != 0 {
            return Err(io::Error::last_os_error());
        }
    }

    Ok(())
}

struct CaptureSource {
    name: String,
    link_type: u16,
    capture: Capture<Active>,
}

fn open_captures() -> Result<Vec<CaptureSource>> {
    let mut captures = Vec::new();

    for device in Device::list().context("failed to list capture devices")? {
        if device.name == "any" || is_bluetooth_interface(&device) {
            continue;
        }

        let name = device.name.clone();
        let mut capture = match Capture::from_device(device)
            .with_context(|| format!("failed to configure capture for {name}"))?
            .promisc(true)
            .immediate_mode(true)
            .snaplen(65_535)
            .timeout(10)
            .open()
        {
            Ok(capture) => capture,
            Err(error) => {
                eprintln!("warning: failed to open capture device {name}: {error}");
                continue;
            }
        };

        if capture.filter(CAPTURE_FILTER, true).is_err() {
            eprintln!("Filter not applicable to {name}");
            continue;
        }

        let capture = match capture.setnonblock() {
            Ok(capture) => capture,
            Err(error) => {
                eprintln!("warning: failed to set capture device {name} nonblocking: {error}");
                continue;
            }
        };

        let link_type = capture.get_datalink().0 as u16;
        if link_type != 1 {
            eprintln!(
                "warning: skipping {name}; link type {link_type} is not Ethernet and albion-network-lib currently extracts Ethernet frames"
            );
            continue;
        }

        captures.push(CaptureSource {
            name,
            link_type,
            capture,
        });
    }

    if captures.is_empty() {
        return Err(anyhow!(
            "no Ethernet capture devices could be opened; check capture privileges"
        ));
    }

    Ok(captures)
}

fn is_bluetooth_interface(device: &Device) -> bool {
    let name = device.name.to_ascii_lowercase();
    let description = device
        .desc
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();

    name.contains("bluetooth")
        || name.starts_with("bluetooth")
        || name.starts_with("bt")
        || description.contains("bluetooth")
}

fn debug_output_packet(decoded: &DecodedPacket) -> Result<()> {
    let should_output = match decoded {
        DecodedPacket::Event(event) => matches!(
            event.code,
            EventCode::JoinedChatChannel
                | EventCode::LeftChatChannel
                | EventCode::NewChatChannels
                | EventCode::RemovedChatChannel
                | EventCode::ChatMessage
        ),

        DecodedPacket::Operation(operation) => false,
        DecodedPacket::Unknown(_) => true,
    };

    if should_output {
        let value = serde_json::to_value(decoded).context("failed to serialize decoded packet")?;

        print_json(&value, true)?;
    }

    Ok(())
}

fn handle_packet(
    parser: &mut PhotonParser,
    packet_number: usize,
    _timestamp_seconds: libc::time_t,
    frame: &[u8],
    link_type: u16,
    host_filter: &HostFilter,
    args: &Args,
    translator_server: Option<&translator::TranslatorServer>,
) -> Result<()> {
    let Some(packet) = extract_udp_payload(frame, Some(link_type)) else {
        return Ok(());
    };

    if !host_filter.contains(packet.source.ip) && !host_filter.contains(packet.destination.ip) {
        return Ok(());
    }

    let before = parser.decoded_packets().len();
    parser
        .receive_packet(
            packet.payload,
            packet_number,
            packet.source,
            packet.destination,
        )
        .ok();

    for decoded in &parser.decoded_packets()[before..] {
        if args.debug {
            debug_output_packet(decoded)?;
        }

        let DecodedPacket::Event(event) = decoded else {
            continue; // Skip non-event packets when --all is specified
        };

        match &event.extracted {
            Some(ExtractedPacket::ChatMessage(message)) => {
                // Detect the language of the message and output the translated
                // if the language was detected as English, just output the message
                let output = match translator_server {
                    Some(translator_server) => translate_chat_message(translator_server, message),
                    None => Cow::Borrowed(message.message.as_str()),
                };

                println!(
                    "[{}][{}] {}: {}",
                    millis_to_time_ampm(message.timestamp),
                    message.channel_type,
                    message.player_name,
                    output
                );
                // print_json(&value, args.pretty)?;
            }
            Some(_) => continue,
            None => continue,
        }
    }

    Ok(())
}

fn translate_chat_message<'a>(
    translator_server: &translator::TranslatorServer,
    message: &'a ChatMessage,
) -> Cow<'a, str> {
    match translator_server.detect_language(&message.message) {
        Ok(detected) if detected.language == "en" || detected.language == "unknown" => {
            Cow::Borrowed(message.message.as_str())
        }
        Ok(detected) => {
            match translator_server.translate_to_english_from(&message.message, &detected.language)
            {
                Ok(translated)
                    if is_usable_translation(&message.message, &translated.translated_text) =>
                {
                    Cow::Owned(translated.translated_text)
                }
                Ok(translated) => {
                    eprintln!(
                        "warning: rejected suspicious {} translation from {} ({} chars -> {} chars)",
                        detected.language,
                        message.player_name,
                        message.message.chars().count(),
                        translated.translated_text.chars().count()
                    );
                    Cow::Borrowed(message.message.as_str())
                }
                Err(error) => {
                    eprintln!(
                        "warning: failed to translate {} chat message from {}: {error}",
                        detected.language, message.player_name
                    );
                    Cow::Borrowed(message.message.as_str())
                }
            }
        }
        Err(error) => {
            eprintln!(
                "warning: failed to detect chat message language from {}: {error}",
                message.player_name
            );
            Cow::Borrowed(message.message.as_str())
        }
    }
}

fn is_usable_translation(original: &str, translated: &str) -> bool {
    let translated_chars = translated.trim().chars().count();
    if translated_chars == 0 {
        return false;
    }

    let original_chars = original.trim().chars().count().max(1);
    let max_translated_chars = original_chars
        .saturating_mul(MAX_TRANSLATION_EXPANSION_FACTOR)
        .max(MIN_TRANSLATION_MAX_CHARS);

    translated_chars <= max_translated_chars
}

pub fn millis_to_time_ampm(timestamp_millis: i64) -> String {
    let dt: DateTime<Local> = Local
        .timestamp_millis_opt(timestamp_millis)
        .single()
        .expect("invalid timestamp millis");

    dt.format("%I:%M %p").to_string()
}

fn print_json(value: &Value, pretty: bool) -> Result<()> {
    if pretty {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string(value)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_reasonably_sized_translation() {
        assert!(is_usable_translation("hola?", "hello?"));
    }

    #[test]
    fn rejects_empty_translation() {
        assert!(!is_usable_translation("hola?", "   "));
    }

    #[test]
    fn rejects_runaway_translation_expansion() {
        let translated = "mainstream".repeat(80);

        assert!(!is_usable_translation("mainstream", &translated));
    }
}
