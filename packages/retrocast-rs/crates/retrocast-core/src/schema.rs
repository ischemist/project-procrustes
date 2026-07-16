//! Validated scalar schema types used at every Rust ownership boundary.

use std::{borrow::Borrow, fmt, ops::Deref};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ScalarError {
    #[error("SMILES must not be empty")]
    EmptySmiles,
    #[error("InChIKey must have the form AAAAAAAAAAAAAA-BBBBBBBBBB-C, got {0:?}")]
    InvalidInchiKey(String),
    #[error("reaction SMILES must not be empty")]
    EmptyReactionSmiles,
    #[error("unsupported RetroCast schema version {0:?}; expected \"2\"")]
    UnsupportedSchemaVersion(String),
}

macro_rules! string_scalar {
    ($name:ident) => {
        impl $name {
            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl Borrow<str> for $name {
            fn borrow(&self) -> &str {
                self.as_str()
            }
        }

        impl Deref for $name {
            type Target = str;

            fn deref(&self) -> &Self::Target {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.as_str() == other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.as_str() == *other
            }
        }

        impl PartialEq<String> for $name {
            fn eq(&self, other: &String) -> bool {
                self.as_str() == other
            }
        }

        impl PartialEq<$name> for String {
            fn eq(&self, other: &$name) -> bool {
                self == other.as_str()
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::try_from(value).map_err(de::Error::custom)
            }
        }
    };
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CanonicalSmiles(String);

impl TryFrom<String> for CanonicalSmiles {
    type Error = ScalarError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err(ScalarError::EmptySmiles)
        } else {
            Ok(Self(value))
        }
    }
}

impl TryFrom<&str> for CanonicalSmiles {
    type Error = ScalarError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

string_scalar!(CanonicalSmiles);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InchiKey(String);

impl TryFrom<String> for InchiKey {
    type Error = ScalarError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let bytes = value.as_bytes();
        let valid = bytes.len() == 27
            && bytes[14] == b'-'
            && bytes[25] == b'-'
            && bytes
                .iter()
                .enumerate()
                .all(|(index, byte)| matches!(index, 14 | 25) || byte.is_ascii_uppercase());
        if valid {
            Ok(Self(value))
        } else {
            Err(ScalarError::InvalidInchiKey(value))
        }
    }
}

impl TryFrom<&str> for InchiKey {
    type Error = ScalarError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

string_scalar!(InchiKey);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReactionSmiles(String);

impl TryFrom<String> for ReactionSmiles {
    type Error = ScalarError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err(ScalarError::EmptyReactionSmiles)
        } else {
            Ok(Self(value))
        }
    }
}

impl TryFrom<&str> for ReactionSmiles {
    type Error = ScalarError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

string_scalar!(ReactionSmiles);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SchemaVersion;

impl SchemaVersion {
    pub const V2: Self = Self;

    pub const fn as_str(self) -> &'static str {
        "2"
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for SchemaVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SchemaVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "2" {
            Ok(Self::V2)
        } else {
            Err(de::Error::custom(ScalarError::UnsupportedSchemaVersion(
                value,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CanonicalSmiles, InchiKey, SchemaVersion};

    #[test]
    fn scalar_types_reject_invalid_wire_values() {
        assert!(serde_json::from_str::<CanonicalSmiles>(r#"""#).is_err());
        assert!(serde_json::from_str::<InchiKey>(r#""not-an-inchikey""#).is_err());
        assert!(serde_json::from_str::<SchemaVersion>(r#""1""#).is_err());
    }

    #[test]
    fn scalar_types_round_trip_as_strings() {
        let key: InchiKey = serde_json::from_str(r#""LFQSCWFLJHTTHZ-UHFFFAOYSA-N""#).unwrap();
        assert_eq!(
            serde_json::to_string(&key).unwrap(),
            r#""LFQSCWFLJHTTHZ-UHFFFAOYSA-N""#
        );
        assert_eq!(serde_json::to_string(&SchemaVersion::V2).unwrap(), r#""2""#);
    }
}
