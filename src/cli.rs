#[derive(Default, Debug)]
pub struct SimConfig {
    pub binary_path: Option<String>,
    pub isa_string: String,
    pub register_inits: Vec<(usize, u32)>,
    pub is_interactive: bool,
    pub is_newlib: bool,
}

impl SimConfig {
    pub fn parse_args(args: &[String]) -> Self {
        let mut config = SimConfig::default();
        let mut i = 0;

        while i < args.len() {
            let arg = &args[i];

            match arg.as_ref() {
                "--newlib" | "-n" => {
                    config.is_newlib = true;
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        config.binary_path = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--isa" => {
                    if i + 1 < args.len() {
                        config.isa_string = args[i + 1].clone();
                        i += 1;
                    }
                }
                setreg if setreg.starts_with("--setreg") => {
                    if let Some(kv) = setreg.strip_prefix("--setreg=") {
                        Self::parse_setreg(&mut config, kv);
                    } else if i + 1 < args.len() {
                        let kv = &args[i + 1];
                        Self::parse_setreg(&mut config, kv);
                        i += 1;
                    }
                }
                "--interactive" => {
                    config.is_interactive = true;
                }
                s if s.starts_with('-') && config.binary_path.is_none() => {
                    let cleaned = arg.trim_start_matches('/').trim_start_matches("working/");
                    config.binary_path = Some(cleaned.to_string());
                }
                _ => {}
            }

            i += 1;
        }

        config
    }

    fn parse_setreg(config: &mut SimConfig, kv: &str) {
        if let Some((reg_name, val_str)) = kv.split_once('=') {
            let reg_idx = match reg_name.trim() {
                "sp" | "x2" => 2,
                "ra" | "x1" => 1,
                "gp" | "x3" => 3,
                "tp" | "x4" => 4,
                "t0" | "x5" => 5,
                "a0" | "x10" => 10,
                "a1" | "x11" => 11,
                "a7" | "x17" => 17,
                s if s.starts_with('x') => s[1..].parse::<usize>().unwrap_or(0),
                _ => 0,
            };

            let val_clean = val_str.trim();
            let val = if val_clean.starts_with("0x") || val_clean.starts_with("0X") {
                u32::from_str_radix(&val_clean[2..], 16).unwrap_or(0)
            } else {
                val_clean.parse::<u32>().unwrap_or(0)
            };

            if reg_idx > 0 && reg_idx < 32 {
                config.register_inits.push((reg_idx, val));
            }
        }
    }
}
