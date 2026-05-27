---
icon: lucide/git-branch
---

# Route Node IDs

RetroCast uses deterministic route-local ids when evaluation annotations need to point back into a `Route`. These ids are not chemical identities; they are addresses inside one serialized route tree.

## Example Route

```mermaid
graph TD
    m_root["rc:m:/ target"] --> r_root["rc:r:/"]
    r_root --> m_0["rc:m:/0 intermediate a"]
    r_root --> m_1["rc:m:/1 intermediate b"]
    r_root --> m_2["rc:m:/2 stock leaf"]

    m_0 --> r_0["rc:r:/0"]
    r_0 --> m_00["rc:m:/0/0 stock leaf"]
    r_0 --> m_01["rc:m:/0/1 stock leaf"]

    m_1 --> r_1["rc:r:/1"]
    r_1 --> m_10["rc:m:/1/0 intermediate c"]
    r_1 --> m_11["rc:m:/1/1 stock leaf"]

    m_10 --> r_10["rc:r:/1/0"]
    r_10 --> m_100["rc:m:/1/0/0 stock leaf"]
```

The same route as a trimmed `Route` tree, with ids shown as annotations:

```json
{
  "target": {
    "id": "rc:m:/",
    "product_of": {
      "id": "rc:r:/",
      "reactants": [
        {
          "id": "rc:m:/0",
          "product_of": {
            "id": "rc:r:/0",
            "reactants": [{ "id": "rc:m:/0/0" }, { "id": "rc:m:/0/1" }]
          }
        },
        {
          "id": "rc:m:/1",
          "product_of": {
            "id": "rc:r:/1",
            "reactants": [
              {
                "id": "rc:m:/1/0",
                "product_of": {
                  "id": "rc:r:/1/0",
                  "reactants": [{ "id": "rc:m:/1/0/0" }]
                }
              },
              { "id": "rc:m:/1/1" }
            ]
          }
        },
        { "id": "rc:m:/2" }
      ]
    }
  }
}
```

## Grammar

```text
route_id  := "rc:" node_kind ":" path
node_kind := "m" | "r"
path      := "/" | "/" index ("/" index)*
index     := nonnegative integer
```

Semantics:

- `rc:m:/` is the target molecule.
- `rc:r:/` is the reaction producing the target molecule.
- `rc:m:/0` is the first reactant molecule of `rc:r:/`.
- `rc:r:/0` is the reaction producing `rc:m:/0`, when that molecule is not a leaf.
- `rc:m:/0/1` is the second reactant molecule under `rc:r:/0`.

The reaction id is always the product molecule path with `rc:r:` instead of `rc:m:`.
Child indices are 0-based and follow the route's canonical reactant list order.
Depth is the number of numeric path segments:

- `rc:m:/` has depth `0`
- `rc:m:/1` has depth `1`
- `rc:m:/1/0` has depth `2`

## Library Access

```python
root_reaction = route.reaction_at("rc:r:/")
print(root_reaction.id(), root_reaction.product().id())

child_reaction = route.reaction_at("rc:r:/1/0")
```

Evaluation annotations use these ids:

```json
{
  "reaction_id": "rc:r:/1/0",
  "validity": {
    "tier 0": { "status": "pass" }
  }
}
```
