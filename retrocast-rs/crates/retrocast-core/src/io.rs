use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

use flate2::{Compression, GzBuilder, read::GzDecoder, write::GzEncoder};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    chem,
    error::{EngineError, Result},
    score::Stocks,
};

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let mut reader = open_reader(path)?;
    let mut payload = String::new();
    reader.read_to_string(&mut payload)?;
    Ok(serde_json::from_str(&payload)?)
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    if path.extension().is_some_and(|extension| extension == "gz") {
        let mut writer = GzEncoder::new(BufWriter::new(file), Compression::default());
        serde_json::to_writer(&mut writer, value)?;
        writer.finish()?;
    } else {
        serde_json::to_writer(BufWriter::new(file), value)?;
    }
    Ok(())
}

/// Write the deterministic, human-readable gzip JSON artifact used by RetroCast.
pub fn write_json_gz(path: &Path, value: &Value) -> Result<()> {
    write_gzip(path, python_pretty_json(value).as_bytes())
}

/// Write deterministic canonical JSONL and return the number of rows.
pub fn write_jsonl_gz(path: &Path, rows: &[Value]) -> Result<usize> {
    let mut payload = Vec::new();
    for row in rows {
        payload.extend_from_slice(crate::provenance::canonical_json(row).as_bytes());
        payload.push(b'\n');
    }
    write_gzip(path, &payload)?;
    Ok(rows.len())
}

pub fn write_lines_gz(path: &Path, lines: &[String]) -> Result<usize> {
    let mut payload = Vec::new();
    for line in lines {
        payload.extend_from_slice(line.as_bytes());
        payload.push(b'\n');
    }
    write_gzip(path, &payload)?;
    Ok(lines.len())
}

pub fn write_csv_gz(path: &Path, rows: &[Vec<String>]) -> Result<usize> {
    let mut payload = Vec::new();
    {
        let mut writer = csv::WriterBuilder::new()
            .terminator(csv::Terminator::CRLF)
            .from_writer(&mut payload);
        for row in rows {
            writer.write_record(row).map_err(csv_error)?;
        }
        writer.flush()?;
    }
    write_gzip(path, &payload)?;
    Ok(rows.len())
}

pub fn read_json_value(path: &Path) -> Result<Value> {
    read_json(path)
}

pub fn read_jsonl_values(path: &Path, skip_empty: bool) -> Result<Vec<Value>> {
    let compressed = path.extension().is_some_and(|extension| extension == "gz");
    let reader: Box<dyn Read> = if compressed {
        Box::new(GzDecoder::new(BufReader::new(File::open(path)?)))
    } else {
        Box::new(BufReader::new(File::open(path)?))
    };
    let mut values = Vec::new();
    for (index, line) in BufReader::new(reader).lines().enumerate() {
        let line = line?;
        let text = line.trim();
        if text.is_empty() {
            if skip_empty {
                continue;
            }
            return Err(EngineError::Jsonl {
                line_number: index + 1,
                message: "empty row".to_owned(),
            });
        }
        values.push(
            serde_json::from_str(text).map_err(|error| EngineError::Jsonl {
                line_number: index + 1,
                message: error.to_string(),
            })?,
        );
    }
    Ok(values)
}

pub fn read_lines_gz(path: &Path) -> Result<Vec<String>> {
    BufReader::new(open_reader(path)?)
        .lines()
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(Into::into)
}

fn write_gzip(path: &Path, payload: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = GzBuilder::new()
        .mtime(0)
        .operating_system(255)
        .write(BufWriter::new(file), Compression::best());
    writer.write_all(payload)?;
    writer.finish()?;
    Ok(())
}

fn python_pretty_json(value: &Value) -> String {
    fn write_value(value: &Value, depth: usize, output: &mut String) {
        match value {
            Value::Array(values) if values.is_empty() => output.push_str("[]"),
            Value::Array(values) => {
                output.push_str("[\n");
                for (index, value) in values.iter().enumerate() {
                    output.push_str(&"  ".repeat(depth + 1));
                    write_value(value, depth + 1, output);
                    if index + 1 != values.len() {
                        output.push(',');
                    }
                    output.push('\n');
                }
                output.push_str(&"  ".repeat(depth));
                output.push(']');
            }
            Value::Object(values) if values.is_empty() => output.push_str("{}"),
            Value::Object(values) => {
                output.push_str("{\n");
                for (index, (key, value)) in values.iter().enumerate() {
                    output.push_str(&"  ".repeat(depth + 1));
                    output.push_str(&crate::provenance::python_json_string(key));
                    output.push_str(": ");
                    write_value(value, depth + 1, output);
                    if index + 1 != values.len() {
                        output.push(',');
                    }
                    output.push('\n');
                }
                output.push_str(&"  ".repeat(depth));
                output.push('}');
            }
            Value::String(value) => output.push_str(&crate::provenance::python_json_string(value)),
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => output.push_str(&value.to_string()),
        }
    }

    let mut output = String::new();
    write_value(value, 0, &mut output);
    output
}

pub fn read_stock(path: &Path, name: &str) -> Result<Stocks> {
    let keys = if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".csv.gz"))
        || path.extension().is_some_and(|extension| extension == "csv")
    {
        read_stock_csv(path)?
    } else {
        read_stock_smiles(path)?
    };
    Ok(BTreeMap::from([(name.to_owned(), keys)]))
}

fn read_stock_csv(path: &Path) -> Result<BTreeSet<String>> {
    let reader = open_reader(path)?;
    let mut csv = csv::Reader::from_reader(reader);
    let headers = csv.headers().map_err(csv_error)?.clone();
    let index = headers
        .iter()
        .position(|header| {
            header.eq_ignore_ascii_case("inchikey") || header.eq_ignore_ascii_case("inchi_key")
        })
        .ok_or_else(|| EngineError::AdapterSchema("stock CSV has no InChIKey column".to_owned()))?;
    let mut keys = BTreeSet::new();
    for row in csv.records() {
        let row = row.map_err(csv_error)?;
        if let Some(value) = row.get(index).filter(|value| !value.is_empty()) {
            keys.insert(value.to_owned());
        }
    }
    Ok(keys)
}

fn read_stock_smiles(path: &Path) -> Result<BTreeSet<String>> {
    let reader = open_reader(path)?;
    let mut keys = BTreeSet::new();
    for line in BufReader::new(reader).lines() {
        let smiles = line?;
        if !smiles.trim().is_empty() {
            keys.insert(chem::normalize(smiles.trim())?.1.into_string());
        }
    }
    Ok(keys)
}

fn open_reader(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.extension().is_some_and(|extension| extension == "gz") {
        Ok(Box::new(GzDecoder::new(BufReader::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn csv_error(error: csv::Error) -> EngineError {
    EngineError::AdapterSchema(format!("stock CSV error: {error}"))
}
