use std::{fmt, num::ParseIntError, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum RoutePath {
    Molecule(Box<[usize]>),
    Reaction(Box<[usize]>),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RoutePathError {
    #[error("route path must start with 'rc:'")]
    MissingPrefix,
    #[error("route path must have form 'rc:<kind>:/...'")]
    InvalidShape,
    #[error("route path kind must be 'm' or 'r'")]
    InvalidKind,
    #[error("route path indices must start with '/'")]
    MissingIndexRoot,
    #[error("route path indices must be canonical non-negative integers")]
    InvalidIndex,
    #[error("only molecule paths have a producing reaction")]
    NotMolecule,
    #[error("only reaction paths have a product molecule or reactants")]
    NotReaction,
}

impl RoutePath {
    pub fn parse(value: &str) -> Result<Self, RoutePathError> {
        value.parse()
    }

    pub fn target() -> Self {
        Self::Molecule(Box::new([]))
    }

    pub fn root_reaction() -> Self {
        Self::Reaction(Box::new([]))
    }

    pub fn indices(&self) -> &[usize] {
        match self {
            Self::Molecule(indices) | Self::Reaction(indices) => indices,
        }
    }

    pub fn depth(&self) -> usize {
        self.indices().len()
    }

    pub fn is_molecule(&self) -> bool {
        matches!(self, Self::Molecule(_))
    }

    pub fn is_reaction(&self) -> bool {
        matches!(self, Self::Reaction(_))
    }

    pub fn produced_by(&self) -> Result<Self, RoutePathError> {
        match self {
            Self::Molecule(indices) => Ok(Self::Reaction(indices.clone())),
            Self::Reaction(_) => Err(RoutePathError::NotMolecule),
        }
    }

    pub fn product(&self) -> Result<Self, RoutePathError> {
        match self {
            Self::Reaction(indices) => Ok(Self::Molecule(indices.clone())),
            Self::Molecule(_) => Err(RoutePathError::NotReaction),
        }
    }

    pub fn reactant(&self, index: usize) -> Result<Self, RoutePathError> {
        match self {
            Self::Reaction(indices) => {
                let mut child = indices.to_vec();
                child.push(index);
                Ok(Self::Molecule(child.into_boxed_slice()))
            }
            Self::Molecule(_) => Err(RoutePathError::NotReaction),
        }
    }
}

#[derive(Clone, Copy)]
enum NodeKind {
    Molecule,
    Reaction,
}

impl FromStr for RoutePath {
    type Err = RoutePathError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let body = value
            .strip_prefix("rc:")
            .ok_or(RoutePathError::MissingPrefix)?;
        let (kind, tail) = body.split_once(':').ok_or(RoutePathError::InvalidShape)?;
        let kind = match kind {
            "m" => NodeKind::Molecule,
            "r" => NodeKind::Reaction,
            _ => return Err(RoutePathError::InvalidKind),
        };
        let indices = tail
            .strip_prefix('/')
            .ok_or(RoutePathError::MissingIndexRoot)?;
        let indices = if indices.is_empty() {
            Vec::new()
        } else {
            indices
                .split('/')
                .map(parse_canonical_index)
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(match kind {
            NodeKind::Molecule => Self::Molecule(indices.into_boxed_slice()),
            NodeKind::Reaction => Self::Reaction(indices.into_boxed_slice()),
        })
    }
}

fn parse_canonical_index(value: &str) -> Result<usize, RoutePathError> {
    let index: usize = value
        .parse()
        .map_err(|_: ParseIntError| RoutePathError::InvalidIndex)?;
    if index.to_string() == value {
        Ok(index)
    } else {
        Err(RoutePathError::InvalidIndex)
    }
}

impl fmt::Display for RoutePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_molecule() { 'm' } else { 'r' };
        write!(formatter, "rc:{kind}:/")?;
        for (position, index) in self.indices().iter().enumerate() {
            if position > 0 {
                formatter.write_str("/")?;
            }
            write!(formatter, "{index}")?;
        }
        Ok(())
    }
}

impl Serialize for RoutePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for RoutePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(de::Error::custom)
    }
}

macro_rules! node_id {
    ($name:ident, $kind:ident, $expected:ident) => {
        #[derive(Clone, Debug, Eq, Hash, PartialEq)]
        pub struct $name(RoutePath);

        impl $name {
            pub fn new(path: RoutePath) -> Result<Self, RoutePathError> {
                if matches!(path, RoutePath::$kind(_)) {
                    Ok(Self(path))
                } else {
                    Err(RoutePathError::$expected)
                }
            }

            pub fn path(&self) -> &RoutePath {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl FromStr for $name {
            type Err = RoutePathError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(RoutePath::parse(value)?)
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.to_string())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                value.parse().map_err(de::Error::custom)
            }
        }
    };
}

node_id!(MoleculeId, Molecule, NotMolecule);
node_id!(ReactionId, Reaction, NotReaction);

#[cfg(test)]
mod tests {
    use super::{MoleculeId, ReactionId, RoutePath, RoutePathError};

    #[test]
    fn parses_and_navigates_canonical_paths() {
        let molecule = RoutePath::parse("rc:m:/1/0").unwrap();
        assert_eq!(molecule.depth(), 2);
        assert_eq!(molecule.produced_by().unwrap().to_string(), "rc:r:/1/0");

        let reaction = RoutePath::parse("rc:r:/1/0").unwrap();
        assert_eq!(reaction.product().unwrap(), molecule);
        assert_eq!(reaction.reactant(2).unwrap().to_string(), "rc:m:/1/0/2");
    }

    #[test]
    fn rejects_noncanonical_indices() {
        assert_eq!(
            RoutePath::parse("rc:m:/01").unwrap_err(),
            RoutePathError::InvalidIndex
        );
        assert!(RoutePath::parse("rc:m:/-1").is_err());
        assert!(RoutePath::parse("rc:m://1").is_err());
    }

    #[test]
    fn node_ids_enforce_their_kind_during_deserialization() {
        let molecule: MoleculeId = serde_json::from_str("\"rc:m:/0\"").unwrap();
        assert_eq!(serde_json::to_string(&molecule).unwrap(), "\"rc:m:/0\"");
        assert!(serde_json::from_str::<ReactionId>("\"rc:m:/0\"").is_err());
    }
}
