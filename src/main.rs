mod cli;
pub mod translator;

use albion_network_lib::{
    DecodedPacket, ExtractedPacket, HostFilter, PhotonParser, extract_udp_payload,
};
use anyhow::{Context, Result, anyhow};
use clap::Parser;
use cli::Args;
use pcap::{Active, Capture, Device};
use serde_json::{Value, json};
use std::{
    i32, io, mem,
    path::Path,
    ptr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const CAPTURE_FILTER: &str = "udp port 5056 or udp port 4535";
const HOSTS_PATH: &str = "hosts.txt";
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
    let args = Args::parse();

    let host_filter = HostFilter::from_file(Path::new(HOSTS_PATH))
        .map_err(|error| anyhow!(error.0))
        .with_context(|| format!("failed to load CIDR ranges from {HOSTS_PATH}"))?;
    let _translator_server =
        translator::TranslatorServer::start().context("failed to start translator sidecar")?;

    run_capture(&host_filter, &args)
}

fn run_capture(host_filter: &HostFilter, args: &Args) -> Result<()> {
    install_shutdown_handlers().context("failed to install shutdown signal handlers")?;

    let mut captures = open_captures()?;

    eprintln!(
        "capturing on {} interfaces with filter {:?} and {HOSTS_PATH}; press Ctrl-C to stop",
        captures.len(),
        CAPTURE_FILTER
    );

    let mut parser = PhotonParser::new("live".to_string(), args.debug);
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

fn handle_packet(
    parser: &mut PhotonParser,
    packet_number: usize,
    timestamp_seconds: libc::time_t,
    frame: &[u8],
    link_type: u16,
    host_filter: &HostFilter,
    args: &Args,
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
            &packet.source.to_string(),
            &packet.destination.to_string(),
        )
        .map_err(|error| anyhow!(error.0))
        .with_context(|| format!("failed to decode packet {packet_number}"))?;

    for decoded in &parser.decoded_packets()[before..] {
        let DecodedPacket::Event(event) = decoded else {
            continue; // Skip non-event packets when --all is specified
        };

        match &event.extracted {
            Some(ExtractedPacket::ChatMessage(message)) => {
                let timestamp = unix_timestamp(timestamp_seconds)
                    .context("failed to convert packet timestamp to Unix timestamp")?;

                let value = json!({
                    "timestamp": timestamp,
                    "source": event.source,
                    "destination": event.destination,
                    "type": "chat_message",
                    "message": message,
                });

                print_json(&value, args.pretty)?;
            },
            Some(_) => continue,
            None => continue,
        }
    }

    Ok(())
}

fn unix_timestamp(seconds: libc::time_t) -> Result<String> {
    let seconds = u64::try_from(seconds).context("packet timestamp was before Unix epoch")?;
    let timestamp = UNIX_EPOCH + Duration::from_secs(seconds);
    let elapsed = timestamp
        .duration_since(SystemTime::UNIX_EPOCH)
        .context("packet timestamp was before Unix epoch")?;
    Ok(elapsed.as_secs().to_string())
}

fn print_json(value: &Value, pretty: bool) -> Result<()> {
    if pretty {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", serde_json::to_string(value)?);
    }
    Ok(())
}
