from uuid import uuid4

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Leaf(Base):
    __tablename__ = "leaf"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    specie_id: Mapped[str] = mapped_column(ForeignKey("specie.id"), nullable=False)
    picture_url: Mapped[str] = mapped_column(String, nullable=False)

    specie: Mapped["Specie"] = relationship(back_populates="leaves")
    features: Mapped[list["LeafFeature"]] = relationship(
        back_populates="leaf", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["LeafPrediction"]] = relationship(back_populates="leaf")

    def get_feature(self, feature_name: str) -> "LeafFeature | None":
        """Retorna a característica com esse nome, se ela estiver cadastrada."""
        return next((feature for feature in self.features if feature.name == feature_name), None)
