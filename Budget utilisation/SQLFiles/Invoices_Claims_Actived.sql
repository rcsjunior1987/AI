SELECT invoices.member_id
<<<<<<< HEAD
     , invoices.invoice_total
     , invoices.state invoice_state
     , claim.invoice_id
     , claim.id claim_id
     , claim.start_date claim_start_date
     , claim.end_date claim_end_date
     , claim.state claim_state
     , claim.funded_amount claim_funded_amount
     , claim.claimed_units
     , claim.claimed_unit_price
     , level3.key
     , level2.key
     , level1.key
  FROM HH_claim claim        
=======
     , invoices.invoice_number
     , invoices.invoice_date
     , invoices.invoice_total
     , invoices.state invoice_state
     , invoices.funded_total
     , invoices.is_reimbursement
     , invoices.submitted_date
     , invoices.approved_date
     , invoices.received_date
     , invoices.updated_at invoice_updated_at
     , invoices.provider_account_id
     , invoices.created_at
     , claim.invoice_id
     , claim.id claim_id
     , claim.claim_key
     , claim.start_date claim_start_date
     , claim.end_date claim_end_date
     , claim.updated_at claim_updated_at
     , claim.unit_price
     , claim.state claim_state
     , claim.invoiced_units
     , claim.unit_price
     , claim.claimed_date
     , claim.funded_date claim_funded_date
     , claim.approved_date
     , claim.item_category_level3_id
     , claim.payment_id
     , claim.received_date
     , claim.submitted_date
     , claim.processed_status
     , claim.invoiced_unit_price
     , claim.funded_amount claim_funded_amount
     , claim.claimed_unit_price
     , level3.key
     , level3.id
     , level3.item_category_level2_id
     , level2.id
     , level2.key
     , level2.name
     , level2.code level2_code
     , level2.item_category_level1_id
     , level1.id
     , level1.key
     , level1.name
     , level1.code level1_code
  FROM HH_claim claim
        
>>>>>>> 8f720b2b4e9cec63e694a4d42aa154bd5f56dde0
  INNER JOIN (
			   SELECT invoice.id
					, invoice.member_id
					, invoice.invoice_number
					, invoice.invoice_date
				    , invoice.invoice_total
					, invoice.state invoice_state
					, invoice.funded_total
				    , invoice.is_reimbursement
					, invoice.submitted_date
					, invoice.approved_date
					, invoice.received_date
				    , invoice.updated_at invoice_updated_at
					, invoice.provider_account_id
                    , invoice.state
                    , invoice.updated_at
                    , invoice.created_at
				 FROM HH_invoice invoice
			     INNER JOIN (
							  SELECT member.id member_id 
								   , MAX(plan.start_date) start_date
								   , plan.end_date
								FROM HH_member member
								INNER JOIN HH_plan plan
								        ON plan.member_key = member.member_key
                          
							 WHERE plan.status = 'PLAN_DELIVERY_ACTIVE'
							GROUP BY plan.member_key
						   ) plans ON plans.member_id = invoice.member_id 
								  AND (Date_Format(invoice.created_at,'%Y-%m-%d') >= Date_Format(plans.start_date,'%Y-%m-%d') AND Date_Format(invoice.created_at,'%Y-%m-%d') <= Date_Format(plans.end_date,'%Y-%m-%d'))
                                  
          ) invoices ON invoices.id = claim.invoice_id
                             
  INNER JOIN HH_item_category_level3 level3
          ON level3.id = claim.item_category_level3_id
          
  INNER JOIN HH_item_category_level2 level2
          ON level2.id = level3.item_category_level2_id
          
  INNER JOIN HH_item_category_level1 level1
          ON level1.id = level2.item_category_level1_id
		
    

   
