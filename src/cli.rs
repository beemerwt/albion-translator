use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about = "Live Albion Online network capture decoder")]
pub struct Args {
    /// Print every decoded Photon packet instead of only extracted Albion models.
    #[arg(long)]
    pub all: bool,

    /// Pretty-print JSON output.
    #[arg(long)]
    pub pretty: bool,

    /// Enable parser debug logging.
    #[arg(long)]
    pub debug: bool,
}
