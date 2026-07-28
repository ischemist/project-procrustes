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

pub fn validate_path_component(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || value.contains('\0')
    {
        return Err(EngineError::UnsafePathComponent {
            label: label.to_owned(),
            value: value.to_owned(),
        });
    }
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(open_reader(path)?)?)
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
    let keys = read_stock_values(path, StockRepresentation::InchiKey)?;
    Ok(BTreeMap::from([(name.to_owned(), keys)]))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StockRepresentation {
    Smiles,
    InchiKey,
}

/// Read the molecule identifiers an external planner needs from a RetroCast stock.
pub fn read_stock_values(
    path: &Path,
    representation: StockRepresentation,
) -> Result<BTreeSet<String>> {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".csv.gz"))
        || path.extension().is_some_and(|extension| extension == "csv")
    {
        let headers: &[&str] = match representation {
            StockRepresentation::Smiles => &["smiles"],
            StockRepresentation::InchiKey => &["inchikey", "inchi_key"],
        };
        return read_stock_csv_column(path, headers);
    }

    let values = read_stock_smiles_lines(path)?;
    match representation {
        StockRepresentation::Smiles => Ok(values),
        StockRepresentation::InchiKey => values
            .into_iter()
            .map(|smiles| Ok(chem::normalize(&smiles)?.1.into_string()))
            .collect(),
    }
}

fn read_stock_csv_column(path: &Path, accepted_headers: &[&str]) -> Result<BTreeSet<String>> {
    let reader = open_reader(path)?;
    let mut csv = csv::Reader::from_reader(reader);
    let headers = csv.headers().map_err(csv_error)?.clone();
    let index = headers
        .iter()
        .position(|header| {
            let header = header.trim();
            accepted_headers
                .iter()
                .any(|expected| header.eq_ignore_ascii_case(expected))
        })
        .ok_or_else(|| {
            EngineError::AdapterSchema(format!(
                "stock CSV has no {} column",
                accepted_headers.join(" or ")
            ))
        })?;
    let mut values = BTreeSet::new();
    for row in csv.records() {
        let row = row.map_err(csv_error)?;
        if let Some(value) = row
            .get(index)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            values.insert(value.to_owned());
        }
    }
    Ok(values)
}

fn read_stock_smiles_lines(path: &Path) -> Result<BTreeSet<String>> {
    let reader = open_reader(path)?;
    let mut smiles_values = BTreeSet::new();
    for line in BufReader::new(reader).lines() {
        let smiles = line?;
        if !smiles.trim().is_empty() {
            smiles_values.insert(smiles.trim().to_owned());
        }
    }
    Ok(smiles_values)
}

pub(crate) fn open_reader(path: &Path) -> Result<Box<dyn Read>> {
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{StockRepresentation, read_stock_values, validate_path_component};

    #[test]
    fn path_components_reject_traversal_and_platform_separators() {
        for value in [
            "",
            ".",
            "..",
            "../outside",
            "folder/name",
            "folder\\name",
            "a\0b",
        ] {
            assert!(
                validate_path_component(value, "stock").is_err(),
                "accepted unsafe path component: {value:?}"
            );
        }
        assert!(validate_path_component("buyables-stock", "stock").is_ok());
    }

    #[test]
    fn stock_reader_selects_smiles_or_inchikey_columns() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("stock.csv");
        fs::write(
            &path,
            "SMILES, InChIKey\n CCO , LFQSCWFLJHTTHZ-UHFFFAOYSA-N \nC,VNWKTOKETHGBQD-UHFFFAOYSA-N\n",
        )
        .unwrap();

        let smiles = read_stock_values(&path, StockRepresentation::Smiles).unwrap();
        let keys = read_stock_values(&path, StockRepresentation::InchiKey).unwrap();

        assert_eq!(smiles.into_iter().collect::<Vec<_>>(), ["C", "CCO"]);
        assert_eq!(
            keys.into_iter().collect::<Vec<_>>(),
            ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "VNWKTOKETHGBQD-UHFFFAOYSA-N"]
        );
    }
}
