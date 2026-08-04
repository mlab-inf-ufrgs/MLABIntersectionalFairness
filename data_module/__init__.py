from .adult import load_and_preprocess_adult
from .compas import load_and_preprocess_compas
from .dropout import load_and_preprocess_dropout
from .intersectional_bias import load_and_preprocess_intersectional_bias
from .loaders_datasus import load_and_preprocess_sih, load_and_preprocess_sim, load_and_preprocess_sinasc
from .loaders_cadunico import load_and_preprocess_cadunico

# Dictionary mapping dataset names to their loader functions and relevant metadata
DATASETS = {
    'Adult 🇺🇸': {
        'loader': load_and_preprocess_adult,
        'target': 'income',
        'favorable_val': 1,
        'protected_attributes': ['sex', 'race', 'age_group', 'education_group', 'relationship'],
        'country': '🇺🇸 Estados Unidos',
        'icon': '💰',
        'domain': 'Censo Demográfico / Renda',
        'link': 'https://archive.ics.uci.edu/dataset/2/adult',
        'year': '1996',
        'n_approx': '~48K',
        'target_label': 'Renda anual',
        'favorable_label': 'Renda > $50K (classe privilegiada)',
        'description': "Avalia se a renda anual excede $50K. Atributos sensíveis incluem sexo, raça e escolaridade. Grupos frequentemente privilegiados: Homens Brancos."
    },
    'COMPAS 🇺🇸': {
        'loader': load_and_preprocess_compas,
        'target': 'two_year_recid',
        'favorable_val': 0,  # Favorable is NOT recidivating
        'protected_attributes': ['sex', 'race', 'age_group'],
        'country': '🇺🇸 Estados Unidos',
        'icon': '⚖️',
        'domain': 'Justiça Criminal',
        'link': 'https://github.com/propublica/compas-analysis',
        'year': '2016',
        'n_approx': '~7K',
        'target_label': 'Reincidência criminal em 2 anos',
        'favorable_label': 'Não reincidência',
        'description': "Estima o risco de reincidência criminal em 2 anos. Atributos sensíveis incluem raça e sexo. O viés estrutural frequentemente superestima o risco para réus afro-americanos."
    },
    'Dropout 🇵🇹': {
        'loader': load_and_preprocess_dropout,
        'target': 'Target',
        'favorable_val': 1,  # Graduate
        'protected_attributes': ['Gender', 'Age_Group', 'Mother_Qualification_Group'],
        'country': '🇵🇹 Portugal',
        'icon': '🎓',
        'domain': 'Educação Superior',
        'link': 'https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success',
        'year': '2022',
        'n_approx': '~4K',
        'target_label': 'Desfecho acadêmico',
        'favorable_label': 'Formatura ou matrícula ativa',
        'description': "Prevê a evasão ou formatura de estudantes universitários. Atributos sensíveis abrangem gênero, idade e qualificação dos pais. Curiosidade: mulheres apresentam taxas naturais de retenção significativamente maiores."
    },
    'Intersectional Bias 🌐': {
        'loader': load_and_preprocess_intersectional_bias,
        'target': 'diagnosis',
        'favorable_val': 1,
        'protected_attributes': ['race', 'sex'],
        'country': '🌐 Internacional',
        'icon': '🔬',
        'domain': 'Saúde (Benchmark Sintético)',
        'link': 'https://www.openml.org/search?type=data&id=44203',
        'year': '',
        'n_approx': '',
        'target_label': 'Diagnóstico clínico',
        'favorable_label': 'Diagnóstico favorável',
        'description': "Desenvolvido especificamente para auditar diagnósticos clínicos com disparidades interseccionais (ex: viés oculto exacerbado na interseção entre raça e sexo)."
    },
    'SIH (DATASUS) 🇧🇷': {
        'loader': load_and_preprocess_sih,
        'target': 'desfecho',
        'favorable_val': 'Alta',
        'protected_attributes': ['sexo', 'raca_cor'],
        'country': '🇧🇷 Brasil',
        'icon': '🏥',
        'domain': 'Saúde Pública — Internações Hospitalares',
        'link': '',
        'year': '',
        'n_approx': '',
        'target_label': 'Desfecho da internação hospitalar',
        'favorable_label': 'Alta hospitalar',
        'description': "Sistema de Informações Hospitalares. Avalia o desfecho da internação (Alta vs. Óbito). Revela disparidades na qualidade da assistência segundo raça/cor e sexo."
    },
    'SIM (DATASUS) 🇧🇷': {
        'loader': load_and_preprocess_sim,
        'target': 'tipo_obito',
        'favorable_val': 'Não Evitável',
        'protected_attributes': ['sexo', 'raca_cor'],
        'country': '🇧🇷 Brasil',
        'icon': '📋',
        'domain': 'Saúde Pública — Mortalidade',
        'link': '',
        'year': '',
        'n_approx': '',
        'target_label': 'Tipo de óbito',
        'favorable_label': 'Óbito não evitável pelo SUS',
        'description': "Sistema de Informações sobre Mortalidade. Analisa se o óbito era evitável por intervenções do SUS. Exibe iniquidades de acesso e mortalidade precoce focada em raça/cor."
    },
    'SINASC (DATASUS) 🇧🇷': {
        'loader': load_and_preprocess_sinasc,
        'target': 'desfecho_nascimento',
        'favorable_val': 'Normal',
        'protected_attributes': ['raca_cor_mae', 'idade_mae', 'escolaridade_mae'],
        'country': '🇧🇷 Brasil',
        'icon': '👶',
        'domain': 'Saúde Materno-Infantil',
        'link': '',
        'year': '',
        'n_approx': '',
        'target_label': 'Desfecho do nascimento',
        'favorable_label': 'Nascimento a termo, sem intercorrências',
        'description': "Sistema de Informações sobre Nascidos Vivos. Foca em desfechos como prematuridade e baixo peso ao nascer, controlando determinantes sociais da mãe."
    },
    'CadÚnico 🇧🇷': {
        'loader': load_and_preprocess_cadunico,
        'target': 'pobreza_extrema',
        'favorable_val': 0,  # Not in extreme poverty
        'protected_attributes': ['raca_cor', 'sexo', 'escolaridade'],
        'country': '🇧🇷 Brasil',
        'icon': '🤝',
        'domain': 'Assistência Social',
        'link': '',
        'year': '2012–2016',
        'n_approx': '',
        'target_label': 'Condição de extrema pobreza',
        'favorable_label': 'Não está em extrema pobreza (renda > R$ 77,00/mês)',
        'description': "Amostra do Cadastro Único. Avalia se a renda per capita familiar está abaixo da linha de extrema pobreza (R$ 77,00 na época). Fortemente marcado pelo racismo e sexismo estrutural."
    }
}
