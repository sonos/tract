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
        if let Ok(m) = Arc::downcast::<InferenceModel>($model.clone()) {
            $expr(m)
        } else if let Ok(m) = Arc::downcast::<TypedModel>($model.clone()) {
            $expr(m)
        } else {
            bail!("Pulse model are unsupported here")
        }
    };
}

#[macro_export]
macro_rules! dispatch_model_mut_no_pulse {
    ($model: expr, $expr: expr) => {
        if let Some(m) = $model.downcast_mut::<InferenceModel>() {
            $expr(m)
        } else if let Some(m) = $model.downcast_mut::<TypedModel>() {
            $expr(m)
        } else {
            bail!("Pulse model are unsupported here")
        }
    };
}
