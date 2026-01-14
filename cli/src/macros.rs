#[macro_export]
macro_rules! dispatch_model {
    ($model: expr, $expr: expr) => {
        (|model: &Arc<dyn Model>| {
            if let Ok(m) = Arc::downcast::<tract_hir::prelude::InferenceModel>(model.clone()) {
                return $expr(m);
            }
            if let Ok(m) = Arc::downcast::<tract_hir::prelude::TypedModel>(model.clone()) {
                return $expr(m);
            }
            #[cfg(feature = "pulse")]
            {
                if let Ok(m) = Arc::downcast::<PulsedModel>(model.clone()) {
                    return $expr(m);
                }
            }
            unreachable!()
        })($model)
    };
}

#[macro_export]
macro_rules! dispatch_model_no_pulse {
    ($model: expr, $expr: expr) => {
        (|model: &Arc<dyn Model>| {
            if let Ok(m) = Arc::downcast::<tract_hir::prelude::InferenceModel>(model.clone()) {
                return $expr(m);
            }
            if let Ok(m) = Arc::downcast::<tract_hir::prelude::TypedModel>(model.clone()) {
                return $expr(m);
            }
            bail!("Pulse model are unsupported here")
        })($model)
    };
}
