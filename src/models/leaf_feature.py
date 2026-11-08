from uuid import uuid4

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class LeafFeature(Base):
    __tablename__ = "leaf_feature"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    leaf_id: Mapped[str] = mapped_column(ForeignKey("leaf.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    leaf: Mapped["Leaf"] = relationship(back_populates="features")

    @staticmethod
    def calculate_feature(leaf_id: str) -> float:
        """Placeholder: implemente o cálculo usando a imagem/dados da folha."""
        raise NotImplementedError("Defina como a característica será calculada.")

    def get_value(self) -> float:
        return self.value
