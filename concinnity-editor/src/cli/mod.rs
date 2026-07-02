// Command-line subcommand implementations for the `concinnity` binary.
//
// This module is declared only by `main.rs`, never by `lib.rs`, so it is
// compiled into the CLI binary and excluded from the `concinnity_editor`
// library. CLI-only std code belongs here; the library stays free of it.

mod add;
mod build;
mod check;
mod explain;
mod list;
mod new;
mod rm;

// Create and apply an asset to the current app
pub use add::add;

// Analyze the current app and report errors, but don't build blob files
pub use check::check;

// Print one asset's effective entry from the expanded world
pub use explain::explain;

// List all declared assets
pub use list::list;

// Create a new app (in the current directory, or a new one)
pub use new::{init, new};

// Delete an asset from the current app
pub use rm::rm;

pub use build::build;
