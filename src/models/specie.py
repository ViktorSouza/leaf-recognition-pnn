from uuid import uuid4

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Specie(Base):
    __tablename__ = "specie"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    picture_url: Mapped[str] = mapped_column(String, nullable=False)

    leaves: Mapped[list["Leaf"]] = relationship(back_populates="specie")

    def get_feature(self, feature_name: str):
        """Retorna a característica da primeira folha que a possui.

        Se as características forem próprias da espécie, modele-as em uma tabela
        separada, em vez de buscá-las nas folhas.
        """
        for leaf in self.leaves:
            feature = leaf.get_feature(feature_name)
            if feature is not None:
                return feature
        return None
