from typing import Annotated, Tuple
from futils.snapshot import Path
from pandantic import Pandantic
import pandas as pd
from pydantic import BaseModel, Field
from binuma import Dataset
from binuma.genotype import BinaryMutationMatrix, DonorGenotype
from binuma.metadata import MetadataDataset
from binuma.sfs import DonorsSfs, DonorsGenotype


class MetadataK(BaseModel):
    donor: str
    sample: str
    age: Annotated[int, Field(int, strict=True, ge=0, le=120)]
    status: str


class MetadataKamizela:
    metadata: pd.DataFrame

    def __init__(self, metadata: pd.DataFrame) -> None:
        validator = Pandantic(schema=MetadataK)
        try:
            self.metadata = validator.validate(dataframe=metadata, errors="raise")
        except ValueError as e:
            e.add_note(f"`metadata` doesn't match its schema\n{metadata}")
            raise e


def load_metadata_kamizela(path2meta: Path) -> MetadataKamizela:
    # per_sample_statistics.n834.csv
    metadata = pd.read_csv(
        path2meta,
        usecols=["Patient", "age_at_sample_exact", "driver3", "tip.label", "status"],
    )
    metadata.rename(
        columns={
            "Patient": "donor",
            "age_at_sample_exact": "age",
            "tip.label": "sample",
        },
        inplace=True,
    )
    metadata["status"] = metadata.status.where(
        ~metadata.driver3.str.contains(r".*BCR::ABL1.*"), "bcr_abl1"
    ).astype(str)
    for col in ["status", "donor", "sample"]:
        metadata[col] = metadata[col].str.lower()
    metadata.drop(columns=["driver3"], inplace=True)
    return MetadataKamizela(metadata)


class DonorsKamizela2025(DonorsGenotype):
    """Mutational matrices from the paper Kamizela 2025 et al. published in
    Nature.

    The matrices are WGS at single-cell resolution of CML patients obtained by
    *in vitro* expansion of individual HSC/HPP cells into clonal colonies
    (clonogenic assay).
    """


def load_kamizela2025_genotype(path2dir: Path) -> DonorsKamizela2025:
    metadata = load_metadata_kamizela(
        path2dir / "per_sample_statistics.n834.csv"
    ).metadata
    donors = list()
    for row in (
        metadata[["donor", "age", "status"]].drop_duplicates().itertuples(index=False)
    ):
        try:
            df = pd.read_csv(path2dir / f"./{row.donor}.csv")
        except FileNotFoundError:
            continue
        # assume nans are 0
        for col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        geno = BinaryMutationMatrix(df)
        donors.append(
            DonorGenotype(
                dataset=Dataset.KAMIZELA2025,
                genotype=geno,
                age=row.age,
                status=row.status,
                name=row.donor,
            )
        )
    print(f"loaded {len(donors)} samples")
    return DonorsKamizela2025(donors)


def load_kamizela2025(path2dir: Path) -> Tuple[DonorsSfs, MetadataDataset]:
    """Load the SFS and the metadata for Kamizela 2025 et al., assuming both
    the SFS and the metadata are in `path2dir`.
    We also assume that the metadata has the following filename: `per_sample_statistics.n834.csv`
    """
    return load_kamizela2025_genotype(path2dir).into_sfs()
